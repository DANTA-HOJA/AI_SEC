import torch
import torchvision
from torch import nn

from ...dataset.imgdataset import NoCropImgDataset_v3
from .resnet50imagetester import ResNet50ImageTester
# -----------------------------------------------------------------------------/


class ResNet50NoCropImageTester(ResNet50ImageTester):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(ResNet50ImageTester, self).__init__(display_on_CLI)
        self._cli_out._set_logger("ResNet50 NoCrop Image Tester")
        
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