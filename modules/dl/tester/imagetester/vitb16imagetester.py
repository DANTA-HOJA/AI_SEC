import torch
import torchvision
from torch import nn

from ...dataset.imgdataset import ImgDataset_v3
from .baseimagetester import BaseImageTester
# -----------------------------------------------------------------------------/


class VitB16ImageTester(BaseImageTester):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 Image Tester")
        
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
        
        self.test_set: ImgDataset_v3 = \
            ImgDataset_v3("test", self.training_config, self.test_df,
                          self.class2num_dict, resize,
                          transform=None, dst_root=self.history_dir,
                          debug_mode=self.debug_mode, display_on_CLI=True)
        # ---------------------------------------------------------------------/


    def _set_model(self):
        """ Load model from torchvision
            - ref: https://github.com/pytorch/vision/issues/7397
        """
        model_construct_fn: function = getattr(torchvision.models, self.model_name)
        
        model_pretrain = self.training_config["model"]["pretrain"]
        self.model: nn.Module = model_construct_fn(weights=model_pretrain)
        
        """ Modify model structure """
        self.model.heads.head = nn.Linear(in_features=768, out_features=len(self.class2num_dict), bias=True)
        self.model.to(self.device)
        
        """ Load `model_state_dict` """
        # model_path = self.history_dir.joinpath(f"{self.model_state}_model.pth")
        # pth_file = torch.load(model_path, map_location=self.device) # unpack to device directly
        # self.model.load_state_dict(pth_file["model_state_dict"])
        
        self._cli_out.write(f"Load model from `torchvision`, "
                            f"name: '{self.model_name}', "
                            f"weights: '{model_pretrain}'")
        # ---------------------------------------------------------------------/


    def _set_loss_fn(self):
        """
        """
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.loss_fn.to(self.device)
        # ---------------------------------------------------------------------/