import torch
from torch import nn
import torchvision

from .baseimagetester import BaseImageTester
from ...dataset.imgdataset import ImgDataset
from ....shared.clioutput import CLIOutput
# -----------------------------------------------------------------------------/


class VitB16ImageTester(BaseImageTester):


    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__()
        self._cli_out = CLIOutput(display_on_CLI, 
                                  logger_name="Vit_B_16 Image Tester")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------/



    def _set_test_set(self):
        """
        """
        resize: int = 224
        
        self.test_set: ImgDataset = \
            ImgDataset("test", self.test_df, self.class2num_dict,
                       resize, self.use_hsv, transform=None,
                       display_on_CLI=True)
        # ---------------------------------------------------------------------/



    def _set_model(self):
        """
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