import os
from glob import glob
from typing import List, Dict, Tuple
from collections import Counter
import logging

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

from imgaug import augmenters as iaa



def set_gpu(cuda_idx:int):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.set_device(cuda_idx)
    device_num = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_num)
    torch.cuda.empty_cache()
    
    return device, device_name



class ImgDataset(Dataset):
    
    def __init__(self, paths:List[str], class_mapper:Dict[str, int], label_in_filename:int, 
                 use_hsv:bool, transform:iaa.Sequential=None, logger:logging.Logger=None):
        
        self.paths = paths
        self.class_mapper = class_mapper
        self.num_classes = len(self.class_mapper)
        self.label_in_filename = label_in_filename
        self.transform = transform
        self.use_hsv = use_hsv
        if logger is not None:
            if self.use_hsv:
                logger.info("※ : using 'HSV' when getting images from the dataset")
            if self.transform is not None:
                logger.info("※ : applying augmentation on the fly")


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        
        path = self.paths[index]

        # read image
        img = cv2.imread(path)
        
        # augmentation on the fly
        if self.transform is not None:
            img = self.transform(image=img)
        
        # choosing the color space, 'RGB' or 'HSV'
        if self.use_hsv: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        else: 
            img = img[:,:,::-1] # BGR -> RGB
        
        # resize to model input size
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = img / 255.0 # normalize to 0~1
        img = np.moveaxis(img, -1, 0) # img_dims == 3: (H, W, C) -> (C, H, W)
        
        
        # read class label
        ## example_name : L_fish_9_A_aug_0NTz2m7j.tiff
        id = path.split(os.sep)[-1].split(".")[0] # filename_without_extension
        cls = id.split("_")[self.label_in_filename]
        cls_idx = self.class_mapper[cls]
        # print(f"image[{index:^4d}] = {path}, class = {all_class[cls_idx]}")

        img = torch.from_numpy(img).float()  # 32-bit float ,  TODO:  choose proper type for fast training, like fp16 or others.
        cls_idx = torch.tensor(cls_idx) # 64-bit int, can be [0] or [1] or [2] only
        
        return img, cls_idx



def caulculate_metrics(log:Dict, average_loss:float,
                       groundtruth_list:List[int], predict_list:List[int],
                       class_mapper:Dict[str, int]):
    
    # Calculate different f1-score
    class_f1 = f1_score(groundtruth_list, predict_list, average=None) # by class
    macro_f1 = f1_score(groundtruth_list, predict_list, average='macro')
    weighted_f1 = f1_score(groundtruth_list, predict_list, average='weighted')
    micro_f1 = f1_score(groundtruth_list, predict_list, average='micro')
    
    # Update 'log'
    ## average_loss
    if average_loss is not None: log["average_loss"] = round(average_loss, 5)
    else: log["average_loss"] = None
    ## f1-score
    for key, value in class_mapper.items(): log[f"{key}_f1"] = round(class_f1[value], 5)
    log["macro_f1"] = round(macro_f1, 5)
    log["weighted_f1"] = round(weighted_f1, 5)
    log["micro_f1"] = round(micro_f1, 5)
    log["average_f1"] = round(((macro_f1 + micro_f1)/2), 5)



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
    # axs[0].set_xticks(list(train_logs["epoch"]))
    axs[0].legend() # turn label on
    axs[0].set_title(loss_key)
    # Score
    axs[1].plot(list(train_logs["epoch"]), list(train_logs[score_key]), label="train")
    axs[1].plot(list(valid_logs["epoch"]), list(valid_logs[score_key]), label="validate")
    # axs[1].set_xticks(list(train_logs["epoch"]))
    axs[1].set_ylim(0.0, 1.1)
    axs[1].legend() # turn label on
    axs[1].set_title(score_key)
    # Close figure
    plt.close()
    # Save figure
    fig_path = os.path.normpath(f"{save_dir}/training_trend.png")
    fig.savefig(fig_path)
    # print("\n", f"figure save @ \n-> {fig_path}\n")



def confusion_matrix_with_class(ground_truth:List[str], prediction:List[str]):
    
    # Count all possible class in ground_truth, prediction
    result_counter = Counter(ground_truth) + Counter(prediction)
    max_count = result_counter.most_common(1)[0][1] # result_counter.most_common(1) -> [('HD', 282)] <class 'list'> (回傳最大的前 x 項)
    all_class = sorted(list(result_counter.keys()))
    
    # Create "confusion matrix"
    confusion_matrix_list = [" "] # 補 (0, 0) 空格
    confusion_matrix_list.extend(all_class) # 加上 column name
    for r_cls in all_class:
        confusion_matrix_list.append(r_cls) # 加上 row name
        for c_cls in all_class:
            match_cnt = 0
            for i in range(len(ground_truth)):
                if (ground_truth[i] == r_cls) and (prediction[i] == c_cls): match_cnt += 1
            confusion_matrix_list.append(match_cnt)
    
    assert len(confusion_matrix_list) == ((len(all_class)+1)**2), 'Create "confusion matrix" failed'
    
    # Show in CLI
    print("Confusion Matrix:\n")
    for enum, item in enumerate(confusion_matrix_list):
        print(f"{item:>{len(str(max_count))+3}}", end="")
        if (enum+1)%(len(all_class)+1) == 0: print("\n")
    print("\n", end="")

    return confusion_matrix_list



def compose_transform() -> iaa.Sequential:
    
    transform = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )),
        # iaa.CropToFixedSize(width=512, height=512),
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
        iaa.Sequential([
            iaa.Sometimes(0.5, [
                iaa.WithChannels([0, 1], iaa.Clouds()), # ch_B, ch_G
                # iaa.Sometimes(0.3, iaa.Cartoon()),
                iaa.GammaContrast((0.5, 2.0)),
                iaa.OneOf([
                    iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
                    iaa.Sharpen(alpha=0.5)
                ])
            ]), 
        ], random_order=True),
        iaa.Dropout2d(p=0.2, nb_keep_channels=2),
    ])
    
    return transform



def calculate_class_weight(class_count_dict:Dict[str, int]) -> torch.Tensor:
    
    """To calculate `class_weight` for Loss function in `List` format

        - how to calculate `class_weight`: https://naadispeaks.wordpress.com/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/
        - applying `class_weight`, ref: https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731 
    
    Args:
        class_count_dict (Dict[str, int]): A `Dict` contains the statistic information for each class, 
        e.g. `{ "L": 450, "M": 740, "S": 800 }`

    Returns:
        torch.tensor: `class_weight` in `torch.Tensor` format
    """
    class_weights_list = []
    total_samples = sum(class_count_dict.values())
    
    for key, value in class_count_dict.items(): # value = number of samples of the class
        class_weights_list.append((1 - (value/total_samples)))

    return torch.tensor(class_weights_list, dtype=torch.float)



def get_sortedClassMapper_from_dir(dir_path) -> Tuple[List[str], Dict[str, int]]:
    
    num2class_list = glob(os.path.normpath(f"{dir_path}/*"))
    num2class_list = [path.split(os.sep)[-1] for path in num2class_list]
    num2class_list.sort()
    class2num_dict = {cls:i for i, cls in enumerate(num2class_list)}
    
    return num2class_list, class2num_dict