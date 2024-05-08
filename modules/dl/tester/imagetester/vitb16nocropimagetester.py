import torch
import torchvision
from torch import nn

from ...dataset.imgdataset import NoCropImgDataset_v3
from .vitb16imagetester import VitB16ImageTester
# -----------------------------------------------------------------------------/


class VitB16NoCropImageTester(VitB16ImageTester):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(VitB16ImageTester, self).__init__(display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 NoCrop Image Tester")
        
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