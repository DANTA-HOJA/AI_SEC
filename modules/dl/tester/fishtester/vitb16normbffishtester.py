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
from ...dataset.imgdataset import NormBFImgDataset_v3
from ...tester.utils import reshape_transform
from .basenormbffishtester import BaseNormBFFishTester
# -----------------------------------------------------------------------------/


class VitB16NormBFFishTester(BaseNormBFFishTester):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(BaseNormBFFishTester, self).__init__(display_on_CLI=display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 NormBF Fish Tester")
        
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
        
        self.test_set: NormBFImgDataset_v3 = \
            NormBFImgDataset_v3("test", self.training_config, self.test_df,
                                self.class2num_dict, resize, self._processed_di,
                                transform=None, dst_root=self.history_dir,
                                debug_mode=self.debug_mode, display_on_CLI=True)
        # ---------------------------------------------------------------------/


    def _set_model(self):
        """ Load model from torchvision
            - ref: https://github.com/pytorch/vision/issues/7397
        """
        model_construct_fn: function = getattr(torchvision.models, self.model_name)
        self.model: nn.Module = model_construct_fn(weights=None)
        
        """ Modify model structure """
        self.model.heads.head = nn.Linear(in_features=768, out_features=len(self.class2num_dict), bias=True)
        self.model.to(self.device)
        
        """ Load `model_state_dict` """
        model_path = self.history_dir.joinpath(f"{self.model_state}_model.pth")
        pth_file = torch.load(model_path, map_location=self.device) # unpack to device directly
        self.model.load_state_dict(pth_file["model_state_dict"])
        
        self._cli_out.write(f"Load model from `torchvision`, "
                            f"name: '{self.model_name}', "
                            f"weights: '{model_path}'")
        # ---------------------------------------------------------------------/


    def _set_loss_fn(self):
        """
        """
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.loss_fn.to(self.device)
        # ---------------------------------------------------------------------/


    def _set_cam_generator(self):
        """ CAM Generator
        
        - (github) ref: https://github.com/jacobgil/pytorch-grad-cam

        - (vit_example) ref: https://github.com/jacobgil/pytorch-grad-cam/blob/2183a9cbc1bd5fc1d8e134b4f3318c3b6db5671f/usage_examples/vit_example.py

        - (Explaination) ref: https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html#how-does-it-work-with-vision-transformers

        # Note
        1. 使用 torch.hub.load('facebookresearch/deit:main','deit_tiny_patch16_224', pretrained=True) 時 target_layers = [model.blocks[-1].norm1]
        2. 透過 print(model) 比較後 `torchvision` 的 `vit_b_16` 應使用 target_layers = [model.encoder.layers.encoder_layer_11.ln_1]
        """
        target_layers = [self.model.encoder.layers.encoder_layer_11.ln_1]
        
        # # 11 LayerNorm (very slow)
        # target_layers = []
        # for i in range(len(self.model.encoder.layers)):
        #     target_layers.append(getattr(self.model.encoder.layers, f"encoder_layer_{i}").ln_1)
        
        self.cam_generator: GradCAM = \
            GradCAM(model=self.model, target_layers=target_layers,
                    use_cuda=True, reshape_transform=reshape_transform)
        
        # self.cam_generator: XGradCAM = \
        #     XGradCAM(model=self.model, target_layers=target_layers,
        #              use_cuda=True, reshape_transform=reshape_transform)
        
        self.cam_generator.batch_size = self.batch_size
        # ---------------------------------------------------------------------/


    # def _set_dff(self):
    #     """
    #     """
    #     target_layers = self.model.encoder.layers.encoder_layer_11
        
    #     self.dff = \
    #         DeepFeatureFactorization(model=self.model,
    #                                  target_layer=target_layers,
    #                                  computation_on_concepts=self.model.heads.head,
    #                                  reshape_transform=reshape_transform)
    #     # ---------------------------------------------------------------------/


    def _set_testing_attrs(self): # extend
        """
        """
        super()._set_testing_attrs()
        
        # get `median` area in BFs
        median_area = self.dataset_df["Trunk surface area, SA (um2)"].median()
        tmp_row = self.dataset_df[self.dataset_df["Trunk surface area, SA (um2)"] == median_area]
        median_fish = list(tmp_row["Brightfield"])[0]
        
        # get image size of `median` area BF
        path = self._processed_di.brightfield_processed_dname_dirs_dict[median_fish]
        img = cv2.imread(str(path.joinpath("Norm_BF.tif")))
        self.img_size = tuple(img.shape[1::-1])
        # ---------------------------------------------------------------------/


    def _save_cam_result(self, crop_name:str, grayscale_cam):
        """
        >>> # overwrite: BaseNormBFFishTester
        """
        
        """ Gray """
        cam_result_dir = self.cam_result_root.joinpath(crop_name, "grayscale_map")
        create_new_dir(cam_result_dir)
        cam_save_path = cam_result_dir.joinpath(f"{crop_name}.graymap.tiff")
        grayscale_cam = np.uint8(255 * grayscale_cam)
        cv2.imwrite(str(cam_save_path), \
                    cv2.resize(grayscale_cam, self.img_size, interpolation=cv2.INTER_LANCZOS4))
        
        """ Color """
        cam_result_dir = self.cam_result_root.joinpath(crop_name, "color_map")
        create_new_dir(cam_result_dir)
        cam_save_path = cam_result_dir.joinpath(f"{crop_name}.colormap.tiff")
        color_cam = cv2.applyColorMap(grayscale_cam,
                                        getattr(cv2, self.colormap)) # BGR
        cv2.imwrite(str(cam_save_path), \
                    cv2.resize(color_cam, self.img_size, interpolation=cv2.INTER_LANCZOS4))
        # ---------------------------------------------------------------------/