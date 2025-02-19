import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from rich import print
from rich.console import Console
from torch.utils.data import Dataset
# -----------------------------------------------------------------------------/



def save_cli_out(dir:Path, cli_out:Console, svg_title="Terminal"):
    """ Save log file in SVG and Text format
    """
    if not isinstance(dir, Path):
        raise TypeError("arg: `dir` is not a `Path` (from pathlib import Path)" )
    # SVG
    cli_out.save_svg(f"{dir}/training_log.svg",
                     title=svg_title, clear=False)
    # Text
    cli_out.save_text(f"{dir}/training_log.log", clear=False)
    # -------------------------------------------------------------------------/



def create_new_dir(dir:Union[str, Path], msg_end:str="\n") -> None:
    """ If `dir` is not exist then create it.

    Args:
        dir (Union[str, Path]): a path
        msg_end (str, optional): control the end of message shows on CLI. Defaults to [NewLine].
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        """ CLI output """
        print(f"Directory: '{dir}' is created!", end=f"{msg_end}")
    # -------------------------------------------------------------------------/



def set_gpu(cuda_idx:int, cli_out:Console=None) -> None:
    """
    """
    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device)
    
    """ CLI output """
    out_msg = f"Using '{device}', device_name = '{device_name}'"
    if cli_out is not None: cli_out.print(out_msg)
    else: print(out_msg)
    
    return device
    # -------------------------------------------------------------------------/



def set_reproducibility(rand_seed):
    """ Pytorch reproducibility
        - ref: https://clay-atlas.com/us/blog/2021/08/24/pytorch-en-set-seed-reproduce/?amp=1
        - ref: https://pytorch.org/docs/stable/notes/randomness.html
    """
    
    """ Seeds """
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed) # current GPU
    torch.cuda.manual_seed_all(rand_seed) # all GPUs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True) # can detect the non-deterministic_algorithms in pytorch
    # -------------------------------------------------------------------------/



def get_exist_bf_dirs(path:Path, glob_pattern:str):
    """
    Find files with `path.glob(glob_pattern)`\n
    the path will exclude if the final part is :
    - (folder) +---delete
    - (file_ext) log
    - (file_ext) toml
    """
    found_list = list(path.glob(glob_pattern))
    i = 0
    while len(found_list) > i:
        dir_name = str(found_list[i]).split(os.sep)[-1]
        if dir_name == "+---delete": found_list.pop(i)
        elif ".log" in dir_name: found_list.pop(i)
        elif ".toml" in dir_name: found_list.pop(i)
        else:
            found_list[i] = found_list[i].parent
            i += 1
    
    return found_list
    # -------------------------------------------------------------------------/



def compose_transform():
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5)
    ])
    
    return transform
    # -------------------------------------------------------------------------/



class BFSegTrainingSet(Dataset):
    
    def __init__(self, path_list:List[Path]) -> None:
        """
        """
        self.path_list: List[Path] = path_list
        self.transform: A.Compose = compose_transform()
        # ---------------------------------------------------------------------/
    
    
    
    def __len__(self):
        """
        """
        return len(self.path_list)
        # ---------------------------------------------------------------------/
    
    
    
    def __getitem__(self, index):
        """
        """
        path: Path = self.path_list[index]
        img_path = path.joinpath("02_cropped_BF.tif")
        seg_path = path.joinpath("Manual_measured_mask.tif")
        
        img: cv2.Mat = cv2.imread(str(img_path), 0)
        seg: cv2.Mat = cv2.imread(str(seg_path), 0)
        
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        seg = cv2.resize(seg, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        transformed = self.transform(image=img, mask=seg)
        img = transformed["image"]
        seg = transformed["mask"]
        
        img = img / 255.0
        seg = seg / 255.0
        
        img = img[np.newaxis, :]
        seg = seg[np.newaxis, :]
        
        img = torch.from_numpy(img).float()
        seg = torch.from_numpy(seg).float()
        
        return str(path), img, seg
        # ---------------------------------------------------------------------/



class BFSegTestSet(Dataset):
    
    def __init__(self, path_list:List[Path]) -> None:
        """
        """
        self.path_list: List[Path] = path_list
        # ---------------------------------------------------------------------/
    
    
    
    def __len__(self):
        """
        """
        return len(self.path_list)
        # ---------------------------------------------------------------------/
    
    
    
    def __getitem__(self, index):
        """
        """
        path: Path = self.path_list[index]
        img_path = path.joinpath("02_cropped_BF.tif")
        
        img: cv2.Mat = cv2.imread(str(img_path), 0)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = img / 255.0
        
        img = img[np.newaxis, :]
        img = torch.from_numpy(img).float()
        
        return str(path), img
        # ---------------------------------------------------------------------/