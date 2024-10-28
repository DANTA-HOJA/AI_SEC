import uuid
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
from tomlkit.toml_document import TOMLDocument
from torch.utils.data import Dataset

from ...data.dataset.utils import drop_too_dark, parse_dataset_file_name
from ...data.processeddatainstance import ProcessedDataInstance
from ...plot.utils import draw_drop_info_on_image
from ...shared.baseobject import BaseObject
from ...shared.utils import create_new_dir
from .augmentation import aug_rotate, dynamic_crop, fake_autofluorescence
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
        self.fake_autofluor = fake_autofluorescence()
        self.use_hsv: bool = config["train_opts"]["data"]["use_hsv"]
        self.random_crop: bool = config["train_opts"]["data"]["random_crop"]
        self.add_bg_class: bool = config["train_opts"]["data"]["add_bg_class"]
        
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
        
        # rotate + random crop
        if (self.mode == "train"):
            img = self.dyn_cropper(image=img) # 只有 img 大於 crop size 時才會 crop
        
        # augmentation on the fly
        if (self.mode == "train") and (self.transform is not None):
            img = self.transform(image=img)
            # img = self.fake_autofluor(image=img)
        
        # get image state ("too dark" detection)
        dark_ratio: float = 0.0
        state: str = ""
        if (self.mode == "train"):
            # realtime calculate
            select, drop = drop_too_dark([img], {"param": self.dataset_param})
            if len(select) > 0:
                dark_ratio = select[0][2]
                state = "preserve"
            if len(drop) > 0:
                dark_ratio = drop[0][2]
                state = "discard"
        else:
            # look up value
            dark_ratio = float(self.df.iloc[index]["dark_ratio"])
            state = self.df.iloc[index]["state"]
        
        
        # adjust pixel value
        img_for_mse = deepcopy(img)
        if (self.mode == "train"):
            if (self.add_bg_class) and (state == "discard"):
                # deprecated
                raise ValueError("Detect error settings in config: "
                                 f"train_opts.data.add_bg_class = {self.add_bg_class}")
                fish_class = "BG"
                img = self._adjust_bright_pixel(img, 0, self.dataset_param["intensity"])
                img_for_mse = self._adjust_bright_pixel(img_for_mse, 255, self.dataset_param["intensity"])
            else:
                img = self._adjust_dark_pixel(img, 0, self.dataset_param["intensity"])
                img_for_mse = self._adjust_dark_pixel(img_for_mse, 255, self.dataset_param["intensity"])
        
        # if self.mode != "train":
        #     assert np.array_equal(img, img_for_mse), "img != img_for_mse"
        
        
        # save meta images (debugging)
        if self.debug_mode:
            self._save_meta_img(img_for_mse, dark_ratio, name, fish_class)
        
        # >>> Prepare images <<<
        img = self._cvt_model_format(img)
        img_for_mse = self._cvt_model_format(img_for_mse)
        
        # >>> Prepare label <<<
        cls_idx: int = self.class2num_dict[fish_class]
        cls_idx = torch.tensor(cls_idx) # To `Tensor` (64-bit int), e.g. [0]
        
        return img, img_for_mse, cls_idx, name
        # ---------------------------------------------------------------------/

    def _cvt_model_format(self, bgr_img:np.ndarray):
        """
        """
        img = bgr_img[:,:,::-1] # BGR -> RGB
        
        # choosing the color model, 'RGB' or 'HSV'
        if self.use_hsv is True: img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL) # BGR -> HSV
        
        # convert format
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
        img = img / 255.0 # normalize to 0~1
        img = np.moveaxis(img, -1, 0) # img_dims == 3: (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).float() # To `Tensor` (32-bit float)
        
        return img
        # ---------------------------------------------------------------------/

    def _save_meta_img(self, bgr_img:np.ndarray, dark_ratio:float,
                       name:str, fish_class:str):
        """
        """
        img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        
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

    def _adjust_dark_pixel(self, bgr_img:np.ndarray, value:int,
                           threshold:int) -> np.ndarray:
        """ change all dark pixel (value <= `threshold`) to another value
        """
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV_FULL)
        ch_brightness = hsv_img[:,:,2]
        mask = ch_brightness <= threshold # create mask
        bgr_img2 = deepcopy(bgr_img)
        bgr_img2[mask, :] = value
        
        return bgr_img2
        # ---------------------------------------------------------------------/

    def _adjust_bright_pixel(self, bgr_img:np.ndarray, value:int,
                             threshold:int) -> np.ndarray:
        """ change all bright pixel (value > `threshold`) to another value
        """
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV_FULL)
        ch_brightness = hsv_img[:,:,2]
        mask = ch_brightness > threshold # create mask
        bgr_img2 = deepcopy(bgr_img)
        bgr_img2[mask, :] = value
        
        return bgr_img2
        # ---------------------------------------------------------------------/



class SurfDGTImgDataset_v3(ImgDataset_v3):

    def __init__(self, mode:str, config:Union[dict, TOMLDocument],
                 df:pd.DataFrame, resize:int, intensity_thres: int, scaler:int,
                 transform:Union[None, iaa.Sequential], dst_root:Path,
                 debug_mode:bool, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(ImgDataset_v3, self).__init__(display_on_CLI)
        self._cli_out._set_logger(f"SurfDGT {mode.capitalize()} Dataset")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.mode = mode
        self.config: Union[dict, TOMLDocument] = config
        self.df: pd.DataFrame = df
        self.resize = (resize, resize)
        self.intensity_thres = intensity_thres
        self.scaler = scaler
        self.transform: Union[None, iaa.Sequential] = transform
        self.dst_root: Path = dst_root.joinpath("debug", self.mode)
        self.debug_mode: bool = debug_mode
        
        self._set_src_root()
        self.use_hsv: bool = config["train_opts"]["data"]["use_hsv"]
        
        # ---------------------------------------------------------------------
        # """ actions """
        
        self.df["scaled_area"] = self.df["area"].apply(lambda x: x/self.scaler)
        
        if self.use_hsv is True:
            self._cli_out.write("※　: using 'HSV' when getting images from the dataset")
        
        if self.transform is not None:
            self._cli_out.write("※　: applying augmentation on the fly")
        
        if self.debug_mode:
            self._cli_out.write("※　: debug mode, all runtime image will save")
            create_new_dir(self.dst_root)
        # ---------------------------------------------------------------------/

    def __getitem__(self, index):
        """
        """
        """ Get name """
        name: str = self.df.iloc[index]["image_name"]
        
        """ Read image """
        path: Path = self.src_root.joinpath(self.df.iloc[index]["path"])
        img: np.ndarray = cv2.imread(str(path))
        area: str = self.df.iloc[index]["scaled_area"]
        
        # >>> Apply different config settings to image <<<
        
        # augmentation on the fly
        if (self.mode == "train") and (self.transform is not None):
            img = self.transform(image=img)
        
        # adjust pixel value
        img_for_mse = deepcopy(img)
        if (self.mode == "train"):
            img = self._adjust_dark_pixel(img, 0, self.intensity_thres)
            img_for_mse = self._adjust_dark_pixel(img_for_mse, 255, self.intensity_thres)
        
        # >>> Prepare images <<<
        img = self._cvt_model_format(img)
        img_for_mse = self._cvt_model_format(img_for_mse)
        
        # >>> Prepare label <<<
        area = torch.tensor(area, dtype=torch.float32) # To `Tensor` (64-bit int), e.g. [0]
        
        return img, img_for_mse, area, name
        # ---------------------------------------------------------------------/



class NoCropImgDataset_v3(ImgDataset_v3):

    def __init__(self, mode:str, config:Union[dict, TOMLDocument],
                 df:pd.DataFrame, class2num_dict:Dict[str, int], resize:int, intensity_thres: int,
                 transform:Union[None, iaa.Sequential], dst_root:Path,
                 debug_mode:bool, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(ImgDataset_v3, self).__init__(display_on_CLI)
        self._cli_out._set_logger(f"NoCrop {mode.capitalize()} Dataset")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.mode = mode
        self.config: Union[dict, TOMLDocument] = config
        self.df: pd.DataFrame = df
        self.class2num_dict: Dict[str, int] = class2num_dict
        self.resize = (resize, resize)
        self.intensity_thres = intensity_thres
        self.transform: Union[None, iaa.Sequential] = transform
        self.dst_root: Path = dst_root.joinpath("debug", self.mode)
        self.debug_mode: bool = debug_mode
        
        self._set_src_root()
        self.use_hsv: bool = config["train_opts"]["data"]["use_hsv"]
        
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
        
        # augmentation on the fly
        if (self.mode == "train") and (self.transform is not None):
            img = self.transform(image=img)
        
        # adjust pixel value
        img_for_mse = deepcopy(img)
        if (self.mode == "train"):
            img = self._adjust_dark_pixel(img, 0, self.intensity_thres)
            img_for_mse = self._adjust_dark_pixel(img_for_mse, 255, self.intensity_thres)
        
        # >>> Prepare images <<<
        img = self._cvt_model_format(img)
        img_for_mse = self._cvt_model_format(img_for_mse)
        
        # >>> Prepare label <<<
        cls_idx: int = self.class2num_dict[fish_class]
        cls_idx = torch.tensor(cls_idx) # To `Tensor` (64-bit int), e.g. [0]
        
        return img, img_for_mse, cls_idx, name
        # ---------------------------------------------------------------------/



class NormBFImgDataset_v3(ImgDataset_v3):

    def __init__(self, mode:str, config:Union[dict, TOMLDocument],
                 df:pd.DataFrame, class2num_dict:Dict[str, int],
                 resize:int, processed_di: ProcessedDataInstance,
                 transform:Union[None, iaa.Sequential],
                 dst_root:Path, debug_mode:bool, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(ImgDataset_v3, self).__init__(display_on_CLI)
        self._cli_out._set_logger(f"Norm BF {mode.capitalize()} Dataset")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.mode = mode
        self.config: Union[dict, TOMLDocument] = config
        self.df: pd.DataFrame = df
        self.class2num_dict: Dict[str, int] = class2num_dict
        self.resize:tuple[int, int] = (resize, resize)
        self.processed_di: ProcessedDataInstance = processed_di
        self.transform: Union[None, iaa.Sequential] = transform
        self.dst_root: Path = dst_root.joinpath("debug", self.mode)
        self.debug_mode: bool = debug_mode
        
        self.use_hsv: bool = config["train_opts"]["data"]["use_hsv"]
        self.aug_rotate = aug_rotate((-90, 90))
        
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

    def __getitem__(self, index):
        """
        """
        """ Get name """
        dname: str = self.df.iloc[index]["Brightfield"]
        
        """ Read image """
        path: Path = self.processed_di.brightfield_processed_dname_dirs_dict[dname]
        # BF
        img: np.ndarray = cv2.imread(str(path.joinpath("Norm_BF.tif")))
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_LANCZOS4)
        # Mask
        mask: np.ndarray = cv2.imread(str(path.joinpath("Norm_Mask.tif")), -1)
        mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_LANCZOS4)
        mask = SegmentationMapsOnImage(mask, shape=img.shape)
        
        """ Get class """
        fish_class: str = self.df.iloc[index]["class"]
        
        # >>> Apply different config settings to image <<<
        
        # rotate
        if (self.mode == "train"):
            img, mask = self.aug_rotate(image=img, segmentation_maps=mask)
        
        # augmentation on the fly
        if (self.mode == "train") and (self.transform is not None):
            img, mask = self.transform(image=img, segmentation_maps=mask)
        
        # adjust pixel value
        img_for_mse = deepcopy(img)
        if (self.mode == "train"):
            img_for_mse[mask.get_arr() < 127] = 127
        
        # >>> Prepare images <<<
        img = self._cvt_model_format(img)
        img_for_mse = self._cvt_model_format(img_for_mse)
        
        # >>> Prepare label <<<
        cls_idx: int = self.class2num_dict[fish_class]
        cls_idx = torch.tensor(cls_idx) # To `Tensor` (64-bit int), e.g. [0]
        
        return img, img_for_mse, cls_idx, dname
        # ---------------------------------------------------------------------/