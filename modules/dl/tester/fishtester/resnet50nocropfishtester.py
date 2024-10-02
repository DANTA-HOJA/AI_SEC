from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from pytorch_grad_cam import (AblationCAM, DeepFeatureFactorization, EigenCAM,
                              EigenGradCAM, FullGrad, GradCAM,
                              GradCAMElementWise, GradCAMPlusPlus, HiResCAM,
                              LayerCAM, ScoreCAM, XGradCAM)
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.image import show_factorization_on_image
from torch import nn

from ....shared.utils import create_new_dir, get_target_str_idx_in_list
from ...dataset.imgdataset import NoCropImgDataset_v3
from ...tester.utils import reshape_transform
from .resnet50fishtester import ResNet50FishTester
# -----------------------------------------------------------------------------/


class ResNet50NoCropFishTester(ResNet50FishTester):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(ResNet50FishTester, self).__init__(display_on_CLI)
        self._cli_out._set_logger("ResNet50 NoCrop Fish Tester")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_test_set(self):
        """
        """
        resize: int = 224
        intensity_thres: int = 30
        
        self.test_set: NoCropImgDataset_v3 = \
            NoCropImgDataset_v3("test", self.training_config, self.test_df,
                                self.class2num_dict, resize, intensity_thres,
                                transform=None, dst_root=self.history_dir,
                                debug_mode=self.debug_mode, display_on_CLI=True)
        # ---------------------------------------------------------------------/


    def _set_testing_attrs(self): # extend
        """
        """
        super()._set_testing_attrs()
        
        tmp_str: str = self.training_config["dataset"]["base_size"]
        tmp_str_list: list[str] = tmp_str.split("_")
        self.img_size: tuple[int, int] = (int(tmp_str_list[0].replace("W", "")),
                                          int(tmp_str_list[1].replace("H", "")))
        # ---------------------------------------------------------------------/


    def _save_cam_result(self, crop_name:str, grayscale_cam): # overwrite
        """
        """
        
        """ Gray """
        cam_result_dir = self.cam_result_root.joinpath(crop_name, "grayscale_map")
        create_new_dir(cam_result_dir)
        cam_save_path = cam_result_dir.joinpath(f"{crop_name+'_graymap'}.tiff")
        grayscale_cam = np.uint8(255 * grayscale_cam)
        cv2.imwrite(str(cam_save_path), \
                    cv2.resize(grayscale_cam, self.img_size, interpolation=cv2.INTER_CUBIC))
        
        """ Color """
        cam_result_dir = self.cam_result_root.joinpath(crop_name, "color_map")
        create_new_dir(cam_result_dir)
        cam_save_path = cam_result_dir.joinpath(f"{crop_name+'_colormap'}.tiff")
        color_cam = cv2.applyColorMap(grayscale_cam,
                                        getattr(cv2, self.colormap)) # BGR
        cv2.imwrite(str(cam_save_path), \
                    cv2.resize(color_cam, self.img_size, interpolation=cv2.INTER_CUBIC))
        # ---------------------------------------------------------------------/