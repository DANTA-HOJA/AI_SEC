import torchvision
from imgaug import augmenters as iaa
from torch import nn, optim

from ..dataset.augmentation import composite_aug
from ..dataset.imgdataset import ImgDataset_v3
from .basetrainer import BaseTrainer
from .utils import calculate_class_weight
# -----------------------------------------------------------------------------/


class VitB16Trainer(BaseTrainer):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 Trainer")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_train_set(self):
        """
        """
        resize: int = 224
        
        if self.aug_on_fly is True: 
            transform: iaa.Sequential = composite_aug()
        else: 
            transform = None
        
        self.train_set: ImgDataset_v3 = \
            ImgDataset_v3("train", self.config, self.train_df,
                          self.class2num_dict, resize, 
                          transform=transform, dst_root=self.dst_root,
                          debug_mode=self.debug_mode, display_on_CLI=True)
        # ---------------------------------------------------------------------/


    def _set_valid_set(self):
        """
        """
        resize: int = 224
        
        self.valid_set: ImgDataset_v3 = \
            ImgDataset_v3("valid", self.config, self.valid_df,
                          self.class2num_dict, resize, 
                          transform=None, dst_root=self.dst_root,
                          debug_mode=self.debug_mode, display_on_CLI=True)
        # ---------------------------------------------------------------------/


    def _set_model(self):
        """ Load model from torchvision
            - ref: https://github.com/pytorch/vision/issues/7397
        """
        model_construct_fn: function = getattr(torchvision.models, self.model_name)
        self.model: nn.Module = model_construct_fn(weights=self.model_pretrain)
        
        """ Modify model structure """
        self.model.heads.head = nn.Linear(in_features=768, out_features=len(self.class2num_dict), bias=True)
        self.model.to(self.device)
        
        self._cli_out.write(f"Load model from `torchvision`, "
                            f"name: '{self.model_name}', "
                            f"pretrain: '{self.model_pretrain}'")
        # ---------------------------------------------------------------------/


    def _set_loss_fn(self):
        """ WARNING: deprecate: (class) BG 是動態產生的，無法得知 class weight
        """
        # if self.forcing_balance is True:
        #     self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        # else: # `loss_function` with `class_weight`
        #     self.loss_fn: nn.CrossEntropyLoss = \
        #         nn.CrossEntropyLoss(weight=calculate_class_weight(self.class_counts_dict))
        
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.loss_fn.to(self.device)
        # ---------------------------------------------------------------------/


    def _set_optimizer(self):
        """
        """
        self.optimizer: optim.AdamW = \
            optim.AdamW(self.model.parameters(),
                        lr=self.lr, weight_decay=self.weight_decay)
        
        # self.optimizer: optim.SGD = \
        #     optim.SGD(self.model.parameters(), momentum=0.9,
        #                 lr=self.lr, weight_decay=self.weight_decay)
        # ---------------------------------------------------------------------/


    def _set_lr_scheduler(self):
        """
        """
        self.lr_scheduler: optim.lr_scheduler.StepLR = \
            optim.lr_scheduler.StepLR(self.optimizer, 
                                      step_size=self.lr_schedular_step, 
                                      gamma=self.lr_schedular_gamma)
        # ---------------------------------------------------------------------/