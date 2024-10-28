import torchvision
from imgaug import augmenters as iaa
from torch import nn, optim

from ..dataset.augmentation import composite_aug
from ..dataset.imgdataset import NormBFImgDataset_v3
from .basenormbftrainer import BaseNormBFTrainer
from .utils import calculate_class_weight
# -----------------------------------------------------------------------------/


class VitB16NormBFTrainer(BaseNormBFTrainer):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI=display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 NormBF Trainer")
        
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
            raise ValueError("Detect error settings in config: "
                             f"train_opts.data.aug_on_fly = {self.aug_on_fly}")
            transform = None
        
        self.train_set: NormBFImgDataset_v3 = \
            NormBFImgDataset_v3("train", self.config, self.train_df,
                                self.class2num_dict, resize, self._processed_di,
                                transform=transform, dst_root=self.dst_root,
                                debug_mode=self.debug_mode, display_on_CLI=True)
        # ---------------------------------------------------------------------/


    def _set_valid_set(self):
        """
        """
        resize: int = 224
        
        self.valid_set: NormBFImgDataset_v3 = \
            NormBFImgDataset_v3("valid", self.config, self.valid_df,
                                self.class2num_dict, resize, self._processed_di,
                                transform=None, dst_root=self.dst_root,
                                debug_mode=self.debug_mode, display_on_CLI=True)
        # ---------------------------------------------------------------------/


    def _set_model(self):
        """ Load model from torchvision
            - ref: https://github.com/pytorch/vision/issues/7397
        """
        model_construct_fn: function = getattr(torchvision.models, self.model_name)
        self.model: nn.Module = model_construct_fn(weights=self.model_pretrain)
        
        # for param in self.model.parameters():
        #     param.requires_grad = False
        
        """ Modify model structure """
        self.model.heads.head = nn.Linear(in_features=768, out_features=len(self.class2num_dict), bias=True)
        self.model.to(self.device)
        
        self._cli_out.write(f"Load model from `torchvision`, "
                            f"name: '{self.model_name}', "
                            f"pretrain: '{self.model_pretrain}'")
        # ---------------------------------------------------------------------/


    def _set_loss_fn(self):
        """
        """
        if self.forcing_balance is True:
            raise ValueError("Detect error settings in config: "
                             f"train_opts.data.forcing_balance = {self.forcing_balance}")
            self.ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        else: # `loss_function` with `class_weight`
            self.ce_loss: nn.CrossEntropyLoss = \
                nn.CrossEntropyLoss(weight=calculate_class_weight(self.class_counts_dict))
        
        self.mse_loss: nn.MSELoss = nn.MSELoss()
        
        self.ce_loss.to(self.device)
        self.mse_loss.to(self.device)
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