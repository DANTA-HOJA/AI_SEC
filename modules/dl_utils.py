import os
from typing import List, Dict

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score



def set_gpu(cuda_idx:int):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.set_device(cuda_idx)
    device_num = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_num)
    torch.cuda.empty_cache()
    
    return device, device_name



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

        # image preprocess
        img = cv2.imread(path)[:,:,::-1] # BGR -> RGB
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) #  CHECK_PT : resize to model input size
        img = img / 255.0 # normalize to 0~1
        img = np.moveaxis(img, -1, 0) # img_info == 3: (H, W, C) -> (C, H, W)
        # TODO:  transfer to HSV domain
        # TODO:  augmentation on fly (optional)
        
        # read class label
        id = path.split(os.sep)[-1].split(".")[0]
        cls = id.split("_")[self.label_in_filename]
        cls_idx = self.class_map[cls]
        # print(f"image[{index:^4d}] = {path}, class = {all_class[cls_idx]}")

        img = torch.from_numpy(img).float()  # 32-bit float ,  TODO:  choose proper type for fast training, like fp16 or others.
        cls_idx = torch.tensor(cls_idx) # 64-bit int, can be [0] or [1] or [2] only
        
        return img, cls_idx



def caulculate_metrics(log:Dict, accum_batch_loss:float, n_batch:int,
                       groundtruth_list:List[int], predict_list:List[int],
                       class_map:Dict[str, int]):
    
    # Calculate average_loss, f1-score for this train_epoch
    avg_loss = accum_batch_loss/n_batch
    class_f1 = f1_score(groundtruth_list, predict_list, average=None) # by class
    macro_f1 = f1_score(groundtruth_list, predict_list, average='macro')
    weighted_f1 = f1_score(groundtruth_list, predict_list, average='weighted')
    micro_f1 = f1_score(groundtruth_list, predict_list, average='micro')
    
    # Create 'log'
    log["avg_loss"] = avg_loss
    for key, value in class_map.items(): log[key] = class_f1[value]
    log["macro_f1"] = macro_f1
    log["weighted_f1"] = weighted_f1
    log["micro_f1"] = micro_f1
    log["average_f1"] = (macro_f1 + micro_f1)/2
    
    return log



def save_model(desc:str, save_dir:str, model_state_dict, optimizer_state_dict, logs):
    
    """
    WARNING:
    
    If you only plan to keep the best performing model (according to the acquired validation loss),
        don't forget that best_model_state = model.state_dict() returns a reference to the state and not its copy!
    
    You must serialize best_model_state or use best_model_state = deepcopy(model.state_dict()) 
        otherwise your best best_model_state will keep getting updated by the subsequent training iterations.
        
    As a result, the final model state will be the state of the overfitted model.
    
    ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    
    """
    
    # composite store
    torch.save({
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "logs": logs
                }, os.path.join(save_dir, f"{desc}_model.pth"))



def plot_training_trend(plt, save_dir:str,
                        loss_key:str, score_key:str,
                        train_logs:pd.DataFrame, valid_logs:pd.DataFrame):
    
    """
        figsize=(w,h) will have
        pixel_x, pixel_y = w*dpi, h*dpi
        e.g.
        (1200,600) pixels => figsize=(15,7.5), dpi= 80
                             figsize=(12,6)  , dpi=100
                             figsize=( 8,4)  , dpi=150
                             figsize=( 6,3)  , dpi=200 etc.
    """
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(14,6), dpi=100)
    fig.suptitle('Training')
    # Loss
    axs[0].plot(list(train_logs["epoch"]), list(train_logs[loss_key]), label="train")
    axs[0].plot(list(valid_logs["epoch"]), list(valid_logs[loss_key]), label="validate")
    axs[0].set_xticks(list(train_logs["epoch"]))
    axs[0].legend() # turn label on
    axs[0].set_title(loss_key)
    # Score
    axs[1].plot(list(train_logs["epoch"]), list(train_logs[score_key]), label="train")
    axs[1].plot(list(valid_logs["epoch"]), list(valid_logs[score_key]), label="validate")
    axs[1].set_xticks(list(train_logs["epoch"]))
    axs[1].set_ylim(0.0, 1.1)
    axs[1].legend() # turn label on
    axs[1].set_title(score_key)
    # Close figure
    plt.close()
    # Save figure
    fig_path = os.path.normpath(f"{save_dir}/training_trend.png")
    fig.savefig(fig_path)
    # print("\n", f"figure save @ \n-> {fig_path}\n")