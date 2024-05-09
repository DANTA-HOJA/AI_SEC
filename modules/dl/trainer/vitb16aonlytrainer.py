import pandas as pd
import torchvision
from imgaug import augmenters as iaa
from torch import nn, optim

from ..dataset.augmentation import composite_aug
from ..dataset.imgdataset import ImgDataset_v3
from .utils import calculate_class_weight
from .vitb16trainer import VitB16Trainer
# -----------------------------------------------------------------------------/


class VitB16AOnlyTrainer(VitB16Trainer):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(VitB16Trainer, self).__init__(display_on_CLI=display_on_CLI)
        self._cli_out._set_logger("Vit_B_16 AOnly Trainer")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_train_valid_df(self):
        """ Set below attributes
            >>> self.train_df: pd.DataFrame
            >>> self.valid_df: pd.DataFrame
        """
        self.train_df: pd.DataFrame = \
                self.dataset_df[(self.dataset_df["dataset"] == "train") & 
                                    (self.dataset_df["fish_pos"] == "A")]
        
        self.valid_df: pd.DataFrame = \
                self.dataset_df[(self.dataset_df["dataset"] == "valid") & 
                                    (self.dataset_df["state"] == "preserve") &
                                        (self.dataset_df["fish_pos"] == "A")]
        
        if self.random_crop:
            self.train_df = self.train_df[(self.train_df["image_size"] == "base")]
        else:
            self.train_df = self.train_df[(self.train_df["image_size"] == "crop")]
        
        # if not self.add_bg_class:
        #     self.train_df = self.train_df[(self.train_df["state"] == "preserve")]
        #     self.valid_df = self.valid_df[(self.valid_df["state"] == "preserve")]
        
        # debug: sampleing for faster speed
        if self.debug_mode:
            self.train_df = self.train_df.sample(n=self.debug_rand_select,
                                                 replace=False,
                                                 random_state=self.rand_seed)
            self.valid_df = self.valid_df.sample(n=self.debug_rand_select,
                                                 replace=False,
                                                 random_state=self.rand_seed)
            self._cli_out.write(f"※　: debug mode, reduce to only {self.debug_rand_select} images")
        # ---------------------------------------------------------------------/