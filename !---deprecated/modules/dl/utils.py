import os
import sys
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Union
from logging import Logger

import torch
import pandas as pd



def set_gpu(cuda_idx:int, logger:Logger=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.set_device(cuda_idx)
    device_num = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_num)
    torch.cuda.empty_cache()
    
    if logger: cli_out = logger.info
    else: cli_out = print
    cli_out(f"Using '{device}', device_name = '{device_name}'")
    
    return device



def gen_class2num_dict(num2class_list:List[str]):
    return {cls:i for i, cls in enumerate(num2class_list)}



def get_fish_path(image_name:str, df_dataset_xlsx:pd.DataFrame):
    
    df_filtered_rows = df_dataset_xlsx[(df_dataset_xlsx['image_name'] == image_name)]
    fish_path = list(df_filtered_rows["path"])[0]
    
    return Path(fish_path)



def get_fish_class(image_name:str, df_dataset_xlsx:pd.DataFrame):
        
    df_filtered_rows = df_dataset_xlsx[(df_dataset_xlsx['image_name'] == image_name)]
    fish_class = list(df_filtered_rows["class"])[0]
    
    return fish_class



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



def rename_training_dir(orig_dir:Path, time_stamp:str, state:str,
                        epochs:int, aug_on_fly:bool, use_hsv:bool):
    
    new_name_desc3 = f"{epochs}_epochs"
    if aug_on_fly: new_name_desc3 += "_AugOnFly"
    if use_hsv: new_name_desc3 += "_HSV"
    new_name = f"{time_stamp}_{{{state}}}_{{{new_name_desc3}}}"
    
    orig_dir_split = str(orig_dir).split(os.sep)
    orig_dir_split[-1] = new_name
    new_dir = os.sep.join(orig_dir_split)
    
    os.rename(orig_dir, new_dir)