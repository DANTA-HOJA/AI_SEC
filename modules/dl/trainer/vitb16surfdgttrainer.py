import torchvision
from imgaug import augmenters as iaa
from torch import nn, optim

from ...shared.utils import log
from ..dataset.augmentation import composite_aug
from ..dataset.imgdataset import SurfDGTImgDataset_v3
from .basesurfdgttrainer import BaseSurfDGTTrainer
from .utils import calculate_class_weight
# -----------------------------------------------------------------------------/


class VitB16SurfDGTTrainer(BaseSurfDGTTrainer):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI=display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 SurfDGT Trainer")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        super()._set_config_attrs()
        
        if self.dataset_file_name != "DS_SURFDGT.csv":
            raise ValueError(f"The expected (config) `dataset.file_name` "
                             f"for `{type(self).__name__}` is "
                             f"'DS_SURFDGT.csv'")
        # ---------------------------------------------------------------------/


    def _set_train_set(self):
        """
        """
        resize: int = 224
        intensity_thres: int = 30
        scaler = 10**int(log(10, self.dataset_df["area"].median()))
        
        if self.aug_on_fly is True: 
            transform: iaa.Sequential = composite_aug()
        else:
            raise ValueError("Detect error settings in config: "
                             f"train_opts.data.aug_on_fly = {self.aug_on_fly}")
            transform = None
        
        self.train_set: SurfDGTImgDataset_v3 = \
            SurfDGTImgDataset_v3("train", self.config, self.train_df,
                                 resize, intensity_thres, scaler,
                                 transform=transform, dst_root=self.dst_root,
                                 debug_mode=self.debug_mode, display_on_CLI=True)
        # ---------------------------------------------------------------------/


    def _set_valid_set(self):
        """
        """
        resize: int = 224
        intensity_thres: int = 30
        scaler = 10**int(log(10, self.dataset_df["area"].median()))
        
        self.valid_set: SurfDGTImgDataset_v3 = \
            SurfDGTImgDataset_v3("valid", self.config, self.valid_df,
                                 resize, intensity_thres, scaler,
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
        self.model.heads.head = nn.Linear(in_features=768, out_features=1, bias=True)
        self.model.to(self.device)
        
        self._cli_out.write(f"Load model from `torchvision`, "
                            f"name: '{self.model_name}', "
                            f"pretrain: '{self.model_pretrain}'")
        # ---------------------------------------------------------------------/


    def _set_loss_fn(self):
        """
        """
        # if self.forcing_balance is True:
        #     raise ValueError("Detect error settings in config: "
        #                      f"train_opts.data.forcing_balance = {self.forcing_balance}")
        #     self.ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        # else: # `loss_function` with `class_weight`
        #     self.ce_loss: nn.CrossEntropyLoss = \
        #         nn.CrossEntropyLoss(weight=calculate_class_weight(self.class_counts_dict))
        
        self.mse_loss_a: nn.MSELoss = nn.MSELoss()
        self.mse_loss: nn.MSELoss = nn.MSELoss()
        
        self.mse_loss_a.to(self.device)
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