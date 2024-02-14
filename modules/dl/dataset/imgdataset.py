import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from imgaug import augmenters as iaa
from PIL import Image
from tomlkit.toml_document import TOMLDocument
from torch.utils.data import Dataset

from ...data.dataset.utils import drop_too_dark, parse_dataset_file_name
from ...plot.utils import draw_drop_info_on_image
from ...shared.baseobject import BaseObject
from ...shared.utils import create_new_dir
from .augmentation import dynamic_crop
# -----------------------------------------------------------------------------/


class ImgDataset_v3(BaseObject, Dataset):

    def __init__(self, mode:str, config:Union[dict, TOMLDocument],
                 df:pd.DataFrame, class2num_dict:Dict[str, int], resize:int,
                 transform:Union[None, iaa.Sequential], dst_root:Path,
                 debug_mode:bool, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger(f"{mode.capitalize()} Dataset")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.mode = mode
        self.config: Union[dict, TOMLDocument] = config
        self.df: pd.DataFrame = df
        self.class2num_dict: Dict[str, int] = class2num_dict
        self.resize = (resize, resize)
        self.transform: Union[None, iaa.Sequential] = transform
        self.dst_root: Path = dst_root.joinpath("debug", self.mode)
        self.debug_mode: bool = debug_mode
        
        self._set_src_root()
        self._set_dataset_param()
        self.crop_size = self.dataset_param["crop_size"]
        self.dyn_cropper = dynamic_crop(self.crop_size)
        self.use_hsv: bool = config["train_opts"]["data"]["use_hsv"]
        self.add_bg_class: bool = config["train_opts"]["data"]["add_bg_class"]
        self.random_crop: bool = config["train_opts"]["data"]["random_crop"]
        
        # ---------------------------------------------------------------------
        # """ actions """
        
        if self.use_hsv is True:
            self._cli_out.write("※　: using 'HSV' when getting images from the dataset")
        
        if self.transform is not None:
            self._cli_out.write("※　: applying augmentation on the fly")
        
        if self.debug_mode:
            self._cli_out.write("※　: debug mode, all runtime image will save")
            create_new_dir(self.dst_root)
        # ---------------------------------------------------------------------/


    def _set_src_root(self) -> None:
        """
        """
        # get config var
        seed_dir: str = self.config["dataset"]["seed_dir"]
        data: str = self.config["dataset"]["data"]
        palmskin_result: str = self.config["dataset"]["palmskin_result"]
        base_size: str = self.config["dataset"]["base_size"]
        
        dataset_cropped: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
        
        self.src_root = \
            dataset_cropped.joinpath(seed_dir, data, palmskin_result, base_size)
        # ---------------------------------------------------------------------/


    def _set_dataset_param(self) -> None:
        """
        """
        name: str = self.config["dataset"]["file_name"]
        self.dataset_param = parse_dataset_file_name(name)
        # ---------------------------------------------------------------------/


    def __len__(self):
        """
        """
        return len(self.df)
        # ---------------------------------------------------------------------/


    def __getitem__(self, index):
        """
        """
        """ Get name """
        name: str = self.df.iloc[index]["image_name"]
        
        """ Read image """
        path: Path = self.src_root.joinpath(self.df.iloc[index]["path"])
        img: np.ndarray = cv2.imread(str(path))
        fish_class: str = self.df.iloc[index]["class"]
        
        # >>> Apply different config settings to image <<<
        
        # random crop
        dark_ratio: float = 0.0
        state: str = ""
        if (self.random_crop) and (self.mode == "train"):
            # random crop
            img = self.dyn_cropper(image=img)
            # detect too dark
            select, drop = drop_too_dark([img], {"param": self.dataset_param})
            if len(select) > 0:
                dark_ratio = select[0][2]
                state = "preserve"
            if len(drop) > 0:
                dark_ratio = drop[0][2]
                state = "discard"
        else:
            dark_ratio = float(self.df.iloc[index]["dark_ratio"])
            state = self.df.iloc[index]["state"]
        
        # replace BG (background) class
        if (self.add_bg_class) and (state == "discard"):
            fish_class = "BG"
        
        # augmentation on the fly
        if (self.transform is not None) and (self.mode == "train"):
            img = self.transform(image=img)
        
        
        """ Choosing the color space, 'RGB' or 'HSV' """
        img = img[:,:,::-1] # BGR -> RGB
        if self.debug_mode: self._save_meta_img(img, dark_ratio, name, fish_class)
        if self.use_hsv is True: img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL) # BGR -> HSV
        
        """ Prepare image """
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
        img = img / 255.0 # normalize to 0~1
        img = np.moveaxis(img, -1, 0) # img_dims == 3: (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).float() # To `Tensor` (32-bit float)
        
        """ Prepare label """
        cls_idx: int = self.class2num_dict[fish_class]
        cls_idx = torch.tensor(cls_idx) # To `Tensor` (64-bit int), e.g. [0]
        
        return img, cls_idx, name
        # ---------------------------------------------------------------------/


    def _save_meta_img(self, rgb_image:np.ndarray, dark_ratio:float,
                       name:str, fish_class:str):
        """
        """
        img = Image.fromarray(rgb_image)
        
        draw_drop_info_on_image(rgb_image=img,
                                intensity=self.dataset_param["intensity"],
                                dark_ratio=dark_ratio,
                                drop_ratio=self.dataset_param["drop_ratio"])
        
        # save image
        if self.mode == "train":
            while True: # using 'UUID Version 1 (Time-based)'
                save_name = f"{fish_class}_{name}_aug_{str(uuid.uuid1().hex)[:8]}.tiff"
                save_path = self.dst_root.joinpath(save_name)
                if not save_path.exists():
                    break
        else:
            save_path = self.dst_root.joinpath(f"{fish_class}_{name}.tiff")
        
        if not save_path.exists():
            img.save(save_path)
        # ---------------------------------------------------------------------/