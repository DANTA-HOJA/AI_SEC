from typing import List, Dict, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from imgaug import augmenters as iaa
import torch
from torch.utils.data import Dataset

from ...shared.clioutput import CLIOutput
# -----------------------------------------------------------------------------/


class ImgDataset(Dataset):
    """ Use `dataset_xlsx` to get the training properties
        
        P.S. Old vesion is using `image_path` directly
    """
    def __init__(self, mode:str, dataset_df:pd.DataFrame, class2num_dict:Dict[str, int],
                 resize:int, use_hsv:bool, transform:Union[None, iaa.Sequential],
                 display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        self._cli_out = CLIOutput(display_on_CLI, 
                                  logger_name=f"{mode.capitalize()} Dataset")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        self.mode: str = mode
        self.dataset_df: pd.DataFrame = dataset_df
        self.class2num_dict: Dict[str, int] = class2num_dict
        self.resize: Tuple[int, int] = (resize, resize) # (W, H), square image, W == H
        self.use_hsv: bool = use_hsv
        self.transform: Union[None, iaa.Sequential] = transform
        
        if self.use_hsv is True: self._cli_out.write("※　: using 'HSV' when getting images from the dataset")
        if self.transform is not None: self._cli_out.write("※　: applying augmentation on the fly")
        # ---------------------------------------------------------------------/



    def __len__(self):
        """
        """
        return len(self.dataset_df)
        # ---------------------------------------------------------------------/



    def __getitem__(self, index):
        """
        """
        """ Get name """
        name = self.dataset_df.iloc[index]["image_name"]
        
        """ Read image """
        fish_path = self.dataset_df.iloc[index]["path"]
        img = cv2.imread(str(fish_path))
        
        """ Augmentation on the fly """
        if self.transform is not None:
            img = self.transform(image=img)
        
        """ Choosing the color space, 'RGB' or 'HSV' """
        if self.use_hsv is True: img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) # BGR -> HSV
        else: img = img[:,:,::-1] # BGR -> RGB
        
        """ Resize and rearrange to the size of model input """
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
        img = img / 255.0 # normalize to 0~1
        img = np.moveaxis(img, -1, 0) # img_dims == 3: (H, W, C) -> (C, H, W)
        
        """ Read class label """
        fish_class = self.dataset_df.iloc[index]["class"]
        cls_idx = self.class2num_dict[fish_class]

        """ To `Tensor` """
        img = torch.from_numpy(img).float()  # 32-bit float
        cls_idx = torch.tensor(cls_idx) # 64-bit int, can be [0] or [1] or [2] only
        
        return img, cls_idx, name
        # ---------------------------------------------------------------------/    