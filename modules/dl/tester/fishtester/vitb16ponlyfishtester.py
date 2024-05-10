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

from ...dataset.imgdataset import ImgDataset_v3
from ...tester.utils import reshape_transform
from .vitb16fishtester import VitB16FishTester
# -----------------------------------------------------------------------------/


class VitB16POnlyFishTester(VitB16FishTester):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(VitB16FishTester, self).__init__(display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 POnly Fish Tester")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_test_df(self):
        """
        """
        self.test_df: pd.DataFrame = \
                self.dataset_df[(self.dataset_df["dataset"] == "test") &
                                    (self.dataset_df["state"] == "preserve") &
                                        (self.dataset_df["fish_pos"] == "P")]
        
        # if not self.add_bg_class:
        #     self.test_df = self.test_df[(self.test_df["state"] == "preserve")]
        
        # debug: sampleing for faster speed
        if self.debug_mode:
            self.test_df = self.test_df.sample(n=self.debug_rand_select,
                                               replace=False,
                                               random_state=self.rand_seed)
            self._cli_out.write(f"※　: debug mode, reduce to only {self.debug_rand_select} images")
        # ---------------------------------------------------------------------/