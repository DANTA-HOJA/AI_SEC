import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
import json

import cv2
import numpy as np
import pandas as pd
import toml
from colorama import Fore, Back, Style
from tqdm.auto import tqdm

from ..data import dname
from ..data.processeddatainstance import ProcessedDataInstance
from ..shared.clioutput import CLIOutput
from ..shared.config import load_config
from ..shared.pathnavigator import PathNavigator
from ..shared.utils import create_new_dir
# -----------------------------------------------------------------------------/


class ImageHorizontalCutter():


    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        self._path_navigator = PathNavigator()
        self.processed_data_instance = ProcessedDataInstance()
        self._cli_out = CLIOutput(display_on_CLI, 
                                  logger_name="Image Horizontal Cutter")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.random_seed: int # self._set_config_attrs()
        self.random_state: np.random.RandomState # self._set_config_attrs()
        self.palmskin_result_alias: str # self._set_config_attrs()
        
        self.save_dir_root: Path # self._set_save_dirs()
        self.save_dir_train: Path # self._set_save_dirs()
        self.save_dir_test: Path # self._set_save_dirs()
        # ---------------------------------------------------------------------/



    def _set_attrs(self, config_file:Union[str, Path]):
        """
        """
        self.processed_data_instance.set_attrs(config_file)
        self._set_config_attrs(config_file)
        self._set_save_dirs()
        
        """ Check images are existing and readable """
        self.processed_data_instance.check_palmskin_images_condition(config_file)
        # ---------------------------------------------------------------------/



    def _set_config_attrs(self, config_file:Union[str, Path]):
        """ Set below attributes
            - `self.random_seed`: int
            - `self.random_state`: np.random.RandomState()
            - `self.palmskin_result_alias`: str
        """
        config = load_config(config_file, cli_out=self._cli_out)
        
        """ [horizontal_cut] """
        self.random_seed: int = config["horizontal_cut"]["random_seed"]
        self.random_state = np.random.RandomState(seed=self.random_seed)
        self.palmskin_result_alias: str = config["horizontal_cut"]["palmskin_result_alias"]
        # ---------------------------------------------------------------------/



    def _set_save_dirs(self):
        """ Set below attributes
            - `self.save_dir_root`: Path
            - `self.save_dir_train`: Path
            - `self.save_dir_test`: Path
        """
        dataset_cropped_v2: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v2")
        
        self.save_dir_root: Path = dataset_cropped_v2.joinpath(f"SEED_{self.random_seed}",
                                                               self.processed_data_instance.instance_name,
                                                               self.palmskin_result_alias)
        self.save_dir_train: Path = self.save_dir_root.joinpath("train")
        self.save_dir_test: Path = self.save_dir_root.joinpath("test")
        # ---------------------------------------------------------------------/



    def run(self, config_file:Union[str, Path]="1.make_dataset.toml"):
        """
        """
        self._cli_out.divide()
        self._set_attrs(config_file)
        
        rand_choice_result = {"up : train, down: test": 0,
                              "up : test, down: train": 0}
        
        xlsx_df: pd.DataFrame = pd.read_excel(self.processed_data_instance.data_xlsx_path, engine='openpyxl')
        palmskin_dnames = sorted(pd.concat([xlsx_df["Palmskin Anterior (SP8)"], xlsx_df["Palmskin Posterior (SP8)"]]), key=dname.get_dname_sortinfo)
        create_new_dir(self.save_dir_train)
        create_new_dir(self.save_dir_test)
        
        """ Check alias and get relative path """
        rel_path = self.processed_data_instance.palmskin_processed_alias_map[self.palmskin_result_alias]
        
        pbar = tqdm(total=len(palmskin_dnames), desc=f"[ Horizontal Cut ] : ")
        for i, palmskin_dname in enumerate(palmskin_dnames):
            
            path = self.processed_data_instance.palmskin_processed_dir.joinpath(palmskin_dname, rel_path)
            img = cv2.imread(str(path))
            assert img.shape[0] == img.shape[1], "Please pad the image to make it a square image."
            assert img.shape[0]%2 == 0, "Image height is not a even number"
            
            """ Horizontal cut ( image -> up, down ) """
            half_position = int(img.shape[0]/2)
            img_up = img[0:half_position, :, :]
            img_down = img[half_position:half_position*2, :, :]
            
            """ Generate `palmskin_dsname` """
            fish_id, fish_pos = dname.get_dname_sortinfo(palmskin_dname)
            palmskin_dsname = f"fish_{fish_id}_{fish_pos}"
            pbar.desc = f"[ Horizontal Cut ] {palmskin_dsname} : "
            pbar.refresh()
            
            """ Control A and P must choose its opposite part as dataset """
            if i%2 == 0: action = self.random_state.choice([True, False], size=1, replace=False)[0]
            else: action = not action
            
            if action:
                """ up : test, down: train
                """
                # if i%2 == 0: tqdm.write("")
                # tqdm.write(f"palmskin_dsname: '{palmskin_dsname}' --> up : test, down: train")
                
                """ Up -> test """
                save_name = f"{palmskin_dsname}_U"
                dir = self.save_dir_test.joinpath(save_name)
                create_new_dir(dir)
                cv2.imwrite(str(dir.joinpath(f"{save_name}.tiff")), img_up)
                
                """ Down -> train """
                save_name = f"{palmskin_dsname}_D"
                dir = self.save_dir_train.joinpath(save_name)
                create_new_dir(dir)
                cv2.imwrite(str(dir.joinpath(f"{save_name}.tiff")), img_down)
                
                rand_choice_result["up : test, down: train"] += 1
                
            else:
                """ up : train, down: test
                """
                # if i%2 == 0: tqdm.write("")
                # tqdm.write(f"palmskin_dsname: '{palmskin_dsname}' --> up : train, down: test")
                
                """ Up -> train """
                save_name = f"{palmskin_dsname}_U"
                dir = self.save_dir_train.joinpath(save_name)
                create_new_dir(dir)
                cv2.imwrite(str(dir.joinpath(f"{save_name}.tiff")), img_up)
                
                """ Down -> test """
                save_name = f"{palmskin_dsname}_D"
                dir = self.save_dir_test.joinpath(save_name)
                create_new_dir(dir)
                cv2.imwrite(str(dir.joinpath(f"{save_name}.tiff")), img_down)
                
                rand_choice_result["up : train, down: test"] += 1
            
            pbar.update(1)
            pbar.refresh()

        pbar.close()
        self._cli_out.write(f"rand_choice_result = {json.dumps(rand_choice_result, indent=4)}")
        # ---------------------------------------------------------------------/