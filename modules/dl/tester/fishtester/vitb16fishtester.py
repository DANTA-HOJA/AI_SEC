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
from .basefishtester import BaseFishTester
# -----------------------------------------------------------------------------/


class VitB16FishTester(BaseFishTester):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 Fish Tester")
        
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