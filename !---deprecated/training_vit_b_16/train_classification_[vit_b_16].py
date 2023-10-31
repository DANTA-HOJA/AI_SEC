import os
import sys
import traceback
from datetime import datetime

from glob import glob
import copy

import argparse
from pathlib import Path
from typing import List, Dict

from tqdm.auto import tqdm

import json
import pandas as pd
import numpy as np

import cv2
import matplotlib.pyplot as plt
import modules.plt_show as plt_show

from sklearn.model_selection import train_test_split
import torch
from torch import nn, utils
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet50, resnext50_32x4d, vit_b_16



def create_new_dir(path:str, end="\n"):
    if not os.path.exists(path):
        # if the demo_folder directory is not exist then create it.
        os.makedirs(path)
        print(f"path: '{path}' is created!{end}")



class ImgDataset(Dataset):
    def __init__(self, paths:List[str], class_map:Dict[str, int], label_in_filename:int):
        self.paths = paths
        self.class_map = class_map
        self.num_classes = len(self.class_map)
        self.label_in_filename = label_in_filename


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        path = self.paths[index]

        # img preprocess
        img = cv2.imread(path)[:,:,::-1] #BGR -> RGB
        #
        # TODO:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) # resize to model input size:224
        #
        img = img / 255.0 # normalize to 0~1
        img = np.moveaxis(img, -1, 0) # img_info == 3: (H, W, C) -> (C, H, W)

        # read class label
        id = path.split(os.sep)[-1].split(".")[0]
        cls = id.split("_")[self.label_in_filename]
        cls_idx = self.class_map[cls]
        # print(f"image[{index:^4d}] = {path}, class = {all_class[cls_idx]}")

        img = torch.from_numpy(img).float()  # 32-bit float
        cls_idx = torch.tensor(cls_idx) # 64-bit int, can be [0] or [1] or [2] only
        
        return img, cls_idx



def save_model(model_state_dict, optimizer_state_dict, logs, model_name:str, save_dir:str, current_epoch:int, time_stamp:str):
    
    """
    NOTE:
    
    If you only plan to keep the best performing model (according to the acquired validation loss),
        don't forget that best_model_state = model.state_dict() returns a reference to the state and not its copy!
    
    You must serialize best_model_state or use best_model_state = deepcopy(model.state_dict()) 
        otherwise your best best_model_state will keep getting updated by the subsequent training iterations.
        
    As a result, the final model state will be the state of the overfitted model.
    
    ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    
    # *** save all informations ***
    save_name = f"{time_stamp}_{model_name}_{current_epoch}.ckpt"
    file_name = os.path.join(save_dir, save_name)
    
    # 複合式儲存
    torch.save({
                "epoch":current_epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "logs":logs
                }, file_name)



def get_args():
    
    parser = argparse.ArgumentParser(description="zebrafish project: classification")
    parser.add_argument(
        "--cuda_idx",
        type=int,
        required=True,
        help="The index of cuda device.",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="The path of the dataset.",
    )
    parser.add_argument(
        "--n_class",
        type=int,
        default=3,
        help="Total classes in the given dataset",
    )
    parser.add_argument(
        "--label_in_filename",
        type=int,
        default=0,
        help="The index of label/class in the filename separate by '_'",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="The ratio to split the dataset.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="model name/architecture.",
    )
    parser.add_argument(
        "--pretrain_weights",
        type=str,
        default="IMAGENET1K_V2",
        help="model's pretrain weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="training args: epoch.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="training args: batch_size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="training args: learning rate.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="path to save model (file_ext = .ckpt, logs embedding).",
    )
    
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    args = get_args()
    
    
    # *** Variable ***
    # args variable
    cuda_idx = args.cuda_idx
    source_dir = args.source_dir
    n_class = args.n_class
    label_in_filename = args.label_in_filename
    train_ratio = args.train_ratio
    model_name = args.model_name
    pretrain_weights = args.pretrain_weights
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    save_dir = args.save_dir
    #
    # variable
    #
    
    
    
    # *** GPU settings ***
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.set_device(cuda_idx)
    #
    device_num = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_num)
    #
    print("="*100, "\n")
    print(f"Using {device}")
    print(f"device_num = {device_num}")
    print(f"device_name = {device_name}", "\n")



    # *** Get dir_name as class/label name ***
    all_class = []
    for file_or_dir in os.listdir(source_dir):
        if os.path.isdir(f"{source_dir}/{file_or_dir}"):
            # print(file_or_dir)
            all_class.append(file_or_dir)
    class_map = {cls:i for i,cls in enumerate(all_class)}
    print(f"class_map = {class_map}\n")



    # *** Get image path ***
    fish_paths = glob(f"{source_dir}/*/*.tiff")
    print(f"total = {len(fish_paths)}")
    #
    # # NOTE: Debug: only select first 200
    # fish_paths = fish_paths[:200]
    # print(f"Debug mode, only select first {len(fish_paths)}\n")



    # *** Split data ***
    fish_path_train, fish_path_test = train_test_split(fish_paths, random_state=2022, train_size=train_ratio)



    # *** Check data dir/file_name and show test image ***
    # train_data 
    print("\n", f"train_data ({len(fish_path_train)})\n", "-"*100)
    [print(f"{i}：img_path = {fish_path_train[i]}") for i in range(5)]
    # test_data
    print("\n", f"test_data ({len(fish_path_test)})\n", "-"*100)
    [print(f"{i}：img_path = {fish_path_test[i]}") for i in range(5)]
    # show images test
    path = fish_path_train[-1]
    print("\n", "*** read test image ***")
    print(f" path = {path}\n")
    os.system("python plt_show.py --window_name {} --img_path {} --rgb".format("\"Test Image\"", path))

    
    
    # *** Create dataSets ***
    train_set = ImgDataset(fish_path_train, class_map=class_map, label_in_filename=label_in_filename)
    val_set = ImgDataset(fish_path_test, class_map=class_map, label_in_filename=label_in_filename)



    # *** Initial dataLoader ***
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print(f"Total_Train_Batches: {train_dataloader.__len__()}\n")
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    print(f"Total_Eval_Batches: {val_dataloader.__len__()}\n")
    
    
    
    # *** Build model ***
    model_name_rp = model_name.replace("-", "_")
    print(f"load model using 'torch.hub.load()', model_name:'{model_name_rp}', pretrain_weights:'{pretrain_weights}'", "\n")
    repo = 'pytorch/vision'
    model = torch.hub.load(repo, model_name_rp, weights=pretrain_weights)
    #
    # print(model.heads.head) # vit_b_16.heads.head --> Linear(in_features=768, out_features=1000, bias=True)
    #
    # modify fc for class = 3 (HD, LD, MD)
    model.heads.head = nn.Linear(in_features=768, out_features=n_class, bias=True)
    #
    print(model)
    #
    model.to(device)
    #
    # os.system("pause")
    
    
    
    # *** Initial loss function and optimizer ***
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    
    
    # *** Create save model directory ***
    time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    save_dir = f"{save_dir}/{time_stamp}"
    create_new_dir(save_dir)
    # print(save_dir)
    
    
    
    # *** Training ***
    # training variable
    logs = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        # 'best_val_loss': 1e10,
        'best_val_acc': 0.0,
    }
    ## best model
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
    best_log = copy.deepcopy(logs)
    ## best log
    best_epoch = 0
    best_time = str
    ## best flag
    FLAG_BEST_VAL = False
    #
    #
    #
    # *** get start_time and start training ***
    for epoch in range(epochs):
        
        # epoch variable
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        print("="*100, f"\nEPOCH：{epoch}\n")
        

        # *** train ***
        model.train() # set to training mode
        for i, data in enumerate(tqdm(train_dataloader, desc="Train: ")): # i = nth batch separate by DataLoader
            x_train, y_train = data
            x_train, y_train = x_train.to(device), y_train.to(device) # move to GPU
            preds = model(x_train)
            loss_value = loss_fn(preds, y_train)
            _, train_pred = torch.max(preds, 1) # get the highest probability

            # update model parameters by back propagation
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad() # clean gradients after step

            # update train metrics 
            train_acc += (train_pred.cpu() == y_train.cpu()).sum().item()
            train_loss += loss_value.item()
            
            
        # update train_loss
        epoch_train_loss = train_loss/len(train_dataloader)
        logs["train_loss"].append(epoch_train_loss)
        
        # update train_acc
        epoch_train_acc = train_acc / len(train_set)
        logs["train_acc"].append(epoch_train_acc)
        
        
        # *** validation ***
        model.eval() # set to evaluation mode
        with torch.no_grad(): 
            for i, data in enumerate(tqdm(val_dataloader, desc="Valid: ")):
                x_val, y_val = data
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val)
                loss_value = loss_fn(preds, y_val)
                _, val_pred = torch.max(preds, 1)

                # update eval metrics
                ## (val_pred.cpu() == y_val.cpu()).sum() => 在一個Batch中，ground_truth = prediciton的總次數 
                ## (val_pred.cpu() == y_val.cpu()).sum().item() => if type(x) = torch.Tensor, type(x.item()) = value(float, int etc.)
                val_acc += (val_pred.cpu() == y_val.cpu()).sum().item()
                val_loss += loss_value.item()


        # update val_loss
        epoch_val_loss = val_loss/len(val_dataloader)
        logs["val_loss"].append(epoch_val_loss)
    
        # update val_acc
        epoch_val_acc = val_acc / len(val_set)
        logs["val_acc"].append(epoch_val_acc)
        # if Best condition
        if epoch_val_acc > logs["best_val_acc"]:
            FLAG_BEST_VAL = True
            logs["best_val_acc"] = epoch_val_acc
            best_epoch = epoch
            best_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            best_log = copy.deepcopy(logs)
        
        
        if FLAG_BEST_VAL == True:
            print("\n☆★☆ BEST_VALIDATION ☆★☆")
            print(f"\nEPOCH：{epoch} => train_loss: {epoch_train_loss:.4f}, train_acc: {epoch_train_acc:.4f} | eval_loss: {epoch_val_loss:.4f} eval_acc: {epoch_val_acc:.4f}\n")
            FLAG_BEST_VAL = False
        else:
            print(f"\nEPOCH：{epoch} => train_loss: {epoch_train_loss:.4f}, train_acc: {epoch_train_acc:.4f} | eval_loss: {epoch_val_loss:.4f} eval_acc: {epoch_val_acc:.4f}\n")
        # sys.exit()



    # *** Save model ***
    ## best
    save_model(best_model_state_dict, best_optimizer_state_dict, 
            best_log, f"best_{model_name}", save_dir, best_epoch, best_time)
    ## final
    time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    save_model(model.state_dict(), optimizer.state_dict(), logs, f"final_{model_name}", save_dir, epochs-1, time_stamp)



    # *** Save log ***
    log_path = f"{save_dir}/{time_stamp}_training_logs.json"
    with open(log_path, "w") as f:
        f.write(json.dumps(logs, indent=2))
    print("\n", f"log save @ \n-> {log_path}\n")
    #
    # show logs
    df = pd.DataFrame(logs)
    print(df, "\n")
    


    # *** Plot training trend ***
    fig = plt.figure("training trend", figsize=(14,6), dpi=100)  # figsize=(w,h) will have
                                                                # pixel_x, pixel_y = w*dpi, h*dpi
                                                                # e.g.
                                                                # (1200,600) pixels => figsize=(15,7.5), dpi= 80
                                                                #                      figsize=(12,6)  , dpi=100
                                                                #                      figsize=( 8,4)  , dpi=150
                                                                #                      figsize=( 6,3)  , dpi=200 etc.
    ## Loss
    fig.add_subplot(1, 2, 1)
    plt.plot(logs['train_loss'])
    plt.plot(logs['val_loss'])
    plt.legend(['train', 'validation'])
    plt.title('Loss')
    ## Accuracy
    fig.add_subplot(1, 2, 2)
    plt.plot(logs['train_acc'])
    plt.plot(logs['val_acc'])
    plt.legend(['train', 'validation'])
    plt.title('Accuracy')
    ## save figure
    fig_path = f"{save_dir}/{time_stamp}_training_trend.png"
    plt.savefig(fig_path)
    print("\n", f"fig save @ \n-> {fig_path}\n")


    print("="*100, "\n", "process all complete !\n")