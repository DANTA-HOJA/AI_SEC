import torch
import torchvision
from torch import nn

from ....shared.utils import log
from ...dataset.imgdataset import SurfDGTImgDataset_v3
from .basesurfdgtimagetester import BaseSurfDGTImageTester
# -----------------------------------------------------------------------------/


class VitB16SurfDGTImageTester(BaseSurfDGTImageTester):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI=display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 SurfDGT Image Tester")
        
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
        scaler = 10**int(log(10, self.dataset_df["area"].median()))
        
        self.test_set: SurfDGTImgDataset_v3 = \
            SurfDGTImgDataset_v3("test", self.training_config, self.test_df,
                                 resize, intensity_thres, scaler,
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
        self.model.heads.head = nn.Linear(in_features=768, out_features=1, bias=True)
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
        self.loss_fn: nn.MSELoss = nn.MSELoss()
        self.loss_fn.to(self.device)
        # ---------------------------------------------------------------------/