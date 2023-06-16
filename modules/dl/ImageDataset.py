import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Union
from logging import Logger

import cv2
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa

import torch
from torch.utils.data import Dataset

abs_module_path = Path("./../../modules/").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from dl.utils import get_fish_class, get_fish_path



class ImgDataset_v2(Dataset):
    """Use `dataset_xlsx` to get training properties ( v1 is direct using `image_name` )
    """
    
    def __init__(self, name_list:List[str], df_dataset_xlsx:pd.DataFrame, class_mapper:Dict[str, int], resize:Tuple[int, int], 
                 use_hsv:bool, transform:iaa.Sequential=None, logger:Logger=None):
        
        self.name_list = name_list
        self.class_mapper = class_mapper
        self.num_classes = len(self.class_mapper)
        self.resize = resize # (W, H)
        self.transform = transform
        self.use_hsv = use_hsv
        self.df_dataset_xlsx = df_dataset_xlsx
        
        if logger: self.cli_out = logger.info
        else: self.cli_out = print
        
        if self.use_hsv: self.cli_out("※ : using 'HSV' when getting images from the dataset")
        if self.transform is not None: self.cli_out("※ : applying augmentation on the fly")


    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, index):
        
        name = self.name_list[index]
        
        # read image
        fish_path = get_fish_path(name, self.df_dataset_xlsx)
        img = cv2.imread(str(fish_path))
        
        # augmentation on the fly
        if self.transform is not None: 
            img = self.transform(image=img)
        
        # choosing the color space, 'RGB' or 'HSV'
        if self.use_hsv: img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        else: img = img[:,:,::-1] # BGR -> RGB
        
        # resize and rearrange to the size of model input
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
        img = img / 255.0 # normalize to 0~1
        img = np.moveaxis(img, -1, 0) # img_dims == 3: (H, W, C) -> (C, H, W)
        
        # read class label
        fish_class = get_fish_class(name, self.df_dataset_xlsx)
        cls_idx = self.class_mapper[fish_class]

        # To `Tensor`
        img = torch.from_numpy(img).float()  # 32-bit float
        cls_idx = torch.tensor(cls_idx) # 64-bit int, can be [0] or [1] or [2] only
        
        return img, cls_idx, name