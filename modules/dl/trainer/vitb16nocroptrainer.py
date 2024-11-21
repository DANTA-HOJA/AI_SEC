import torchvision
from imgaug import augmenters as iaa
from torch import nn, optim

from ..dataset.augmentation import composite_aug
from ..dataset.imgdataset import NoCropImgDataset_v3
from .vitb16trainer import VitB16Trainer
from .utils import calculate_class_weight
# -----------------------------------------------------------------------------/


class VitB16NoCropTrainer(VitB16Trainer):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(VitB16Trainer, self).__init__(display_on_CLI=display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 NoCrop Trainer")
        
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
        super(VitB16Trainer, self)._set_config_attrs()
        
        if self.dataset_file_name != "DS_SURF3C_NOCROP.csv":
            raise ValueError(f"The expected (config) `dataset.file_name` "
                             f"for `{type(self).__name__}` is "
                             f"'DS_SURF3C_NOCROP.csv'")
        # ---------------------------------------------------------------------/


    def _set_train_set(self):
        """
        """
        resize: int = 224
        intensity_thres: int = 30
        
        if self.aug_on_fly is True: 
            transform: iaa.Sequential = composite_aug()
        else:
            raise ValueError("Detect error settings in config: "
                             f"train_opts.data.aug_on_fly = {self.aug_on_fly}")
            transform = None
        
        self.train_set: NoCropImgDataset_v3 = \
            NoCropImgDataset_v3("train", self.config, self.train_df,
                                self.class2num_dict, resize, intensity_thres,
                                transform=transform, dst_root=self.dst_root,
                                debug_mode=self.debug_mode, display_on_CLI=True)
        # ---------------------------------------------------------------------/


    def _set_valid_set(self):
        """
        """
        resize: int = 224
        intensity_thres: int = 30
        
        self.valid_set: NoCropImgDataset_v3 = \
            NoCropImgDataset_v3("valid", self.config, self.valid_df,
                                self.class2num_dict, resize, intensity_thres,
                                transform=None, dst_root=self.dst_root,
                                debug_mode=self.debug_mode, display_on_CLI=True)
        # ---------------------------------------------------------------------/