import copy
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from rich import print
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

import models
from loss import dice_loss
from utils import (BFSegTrainingSet, create_new_dir, get_exist_bf_dirs,
                   save_cli_out, set_gpu, set_reproducibility)

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, get_repo_root

console = Console(record=True)

install()
# -----------------------------------------------------------------------------/



def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target) # 自帶 sigmoid
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    hybrid_loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['hybrid'] += hybrid_loss.data.cpu().numpy() * target.size(0)
    
    return hybrid_loss
    # -------------------------------------------------------------------------/


def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:.6f}".format(k, metrics[k] / epoch_samples))
    
    console.print("{}_loss: {}".format(phase, ", ".join(outputs)))
    # -------------------------------------------------------------------------/


def train_model(path_navigator: PathNavigator,
                dataloaders:dict[str, DataLoader],
                model_name:str, model, optimizer, scheduler,
                num_epochs:int=25):
    """
    """
    time_stamp: str = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    bfseg_model_root: Path = \
        path_navigator.dbpp.get_one_of_dbpp_roots("model_bfseg")
    history_dir = bfseg_model_root.joinpath(f"{time_stamp}_{{{num_epochs}_epochs}}")
    create_new_dir(history_dir)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        console.print('\n[cyan]Epoch {}/{}'.format(epoch+1, num_epochs))
        console.print('-' * 10)
        
        since = time.time() # timer for single epoch

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    console.print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
            # batch st.
            for paths, inputs, labels in track(dataloaders[phase],
                                               description=f"[yellow]{phase}:",
                                               transient=True):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
            
            assert epoch_samples == len(dataloaders[phase].dataset.path_list)
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['hybrid'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                console.print("[green]saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('[yellow]{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    console.print('\n[magenta]Best val loss: {:4f}'.format(best_loss))

    torch.save(best_model_wts, f"{history_dir}/{model_name}_best.pth")
    save_cli_out(history_dir, console, f"Training_log : {time_stamp}")
    console.print("[green]Done! \n")
    # -------------------------------------------------------------------------/



if __name__ == "__main__":

    print(f"Repository: '{get_repo_root()}'")

    """ Init components """
    path_navigator = PathNavigator()
    processed_di = ProcessedDataInstance()
    processed_di.parse_config("bf_seg.toml")
    
    batch_size: int = 16
    model_name: str = "res18unet"
    device = set_gpu(0, console)
    set_reproducibility(2022)

    # get paths, split into 3 list (train, val, test)
    # -> path example: ".../{Data}_Processed/{20230827_test}_Academia_Sinica_i505/{autothres_triangle}_BrightField_analyze"
    path = processed_di.brightfield_processed_dir
    found_list = get_exist_bf_dirs(path, "*/Manual_measured_mask.tif")
    if found_list == []:
        raise ValueError("Can't find any directories. Make sure that `data_processed.instance_desc` exists.")
    print(f"Found {len(found_list)} directories")
    
    # training_list, test_list = train_test_split(found_list, test_size=0.1, random_state=2022)
    # train_list, valid_list = train_test_split(training_list, test_size=0.2, random_state=2022)
    train_list, valid_list = train_test_split(found_list, test_size=0.1, random_state=2022)
    
    # dataset, dataloader
    train_set = BFSegTrainingSet(train_list)
    val_set = BFSegTrainingSet(valid_list)
    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    # model
    num_class = 1
    if model_name == "unet":
        model = models.UNet(num_class).to(device)
    elif model_name == "res18unet":
        model = models.ResNetUNet(num_class).to(device)
    else:
        raise NotImplementedError

    # Observe that all parameters are being optimized
    optimizer_ft = optim.AdamW(model.parameters(), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

    train_model(path_navigator, dataloaders, model_name, model,
                optimizer_ft, exp_lr_scheduler, num_epochs=50)
    # -------------------------------------------------------------------------/