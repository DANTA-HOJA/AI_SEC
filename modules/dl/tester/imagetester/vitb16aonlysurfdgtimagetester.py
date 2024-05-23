import pandas as pd
import torch
import torchvision
from torch import nn

from ....shared.utils import log
from ...dataset.imgdataset import SurfDGTImgDataset_v3
from .vitb16surfdgtimagetester import VitB16SurfDGTImageTester
# -----------------------------------------------------------------------------/


class VitB16AOnlySurfDGTImageTester(VitB16SurfDGTImageTester):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(VitB16SurfDGTImageTester, self).__init__(display_on_CLI=display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 AOnly SurfDGT Image Tester")
        
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
                                        (self.dataset_df["fish_pos"] == "A")]
        
        # if not self.add_bg_class:
        #     self.test_df = self.test_df[(self.test_df["state"] == "preserve")]
        
        # debug: sampleing for faster speed
        if self.debug_mode:
            self.test_df = self.test_df.sample(n=self.debug_rand_select,
                                               replace=False,
                                               random_state=self.rand_seed)
            self._cli_out.write(f"※　: debug mode, reduce to only {self.debug_rand_select} images")
        # ---------------------------------------------------------------------/