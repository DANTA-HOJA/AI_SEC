import torch
from torch import nn
import torchvision
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, \
    GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

from .basefishtester import BaseFishTester
from ...tester.utils import reshape_transform
from ...dataset.imgdataset import ImgDataset
from ....shared.clioutput import CLIOutput
# -----------------------------------------------------------------------------/


class VitB16FishTester(BaseFishTester):


    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__()
        self._cli_out = CLIOutput(display_on_CLI, 
                                  logger_name="Vit_B_16 Fish Tester")
        
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
        2. 透過 print(model) 比較後 `torchvision` 的 `vit_b_16` 應使用 target_layers = [model.encoder.layers.encoder_layer_10.ln_1]
        """
        target_layers = [self.model.encoder.layers.encoder_layer_10.ln_1]
        
        self.cam_generator: GradCAM = \
            GradCAM(model=self.model, target_layers=target_layers,
                    use_cuda=True, reshape_transform=reshape_transform)
        
        self.cam_generator.batch_size = self.batch_size
        
        self._cli_out.write(f"Do CAM, colormap using '{self.colormap_key}'")
        # ---------------------------------------------------------------------/