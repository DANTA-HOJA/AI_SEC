import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
import shutil

import cv2
from colorama import Fore, Back, Style
from tqdm.auto import tqdm

from . import dsname
from .utils import gen_dataset_xlsx_name, gen_crop_img
from ..data.processeddatainstance import ProcessedDataInstance
from ..shared.clioutput import CLIOutput
from ..shared.config import load_config
from ..shared.pathnavigator import PathNavigator
from ..shared.utils import create_new_dir, formatter_padr0
# -----------------------------------------------------------------------------/


class ImageCropper():


    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        self._path_navigator = PathNavigator()
        self.processed_data_instance = ProcessedDataInstance()
        self._cli_out = CLIOutput(display_on_CLI, 
                                  logger_name="Image Cropper")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        self.config: Union[str, Path] # self._set_attrs()
        self.palmskin_result_alias: str # self._set_config_attrs()
        self.random_seed: int # self._set_config_attrs()
        self.crop_size: int # self._set_config_attrs()
        self.shift_region: str # self._set_config_attrs()
        
        self.save_dir_root: Path # self._set_save_dirs()
        self.save_dir_train: Path # self._set_save_dirs()
        self.save_dir_test: Path # self._set_save_dirs()
        
        self.crop_dir_name: str # self._set_crop_dir_name()
        # ---------------------------------------------------------------------/



    def _set_attrs(self, config_file:Union[str, Path]):
        """
        """
        self.config = load_config(config_file, cli_out=self._cli_out)
        self.processed_data_instance.set_attrs(config_file)
        self._set_config_attrs()
        self._set_save_dirs()
        self._set_crop_dir_name()
        # ---------------------------------------------------------------------/



    def _set_config_attrs(self):
        """ Set below attributes
            - `self.palmskin_result_alias`: str
            - `self.random_seed`: int
            - `self.crop_size`: int
            - `self.shift_region`: str
        """
        """ [data_processed] """
        self.palmskin_result_alias: str = self.config["data_processed"]["palmskin_result_alias"]
        
        """ [param] """
        self.random_seed: int = self.config["param"]["random_seed"]
        self.crop_size: int = self.config["param"]["crop_size"]
        self.shift_region: str = self.config["param"]["shift_region"]
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
        
        self._check_if_save_dirs_exist()
        # ---------------------------------------------------------------------/



    def _check_if_save_dirs_exist(self):
        """
        """
        for dir in [self.save_dir_root, self.save_dir_train, self.save_dir_test]:
            if not dir.exists():
                raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find directories, "
                                        f"run `1.1.horizontal_cut.py` before crop. {Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/



    def _set_crop_dir_name(self):
        """
        """
        temp_dict = gen_dataset_xlsx_name(self.config, dict_format=True)
        
        self.crop_dir_name: str = f"{temp_dict['crop_size']}_{temp_dict['shift_region']}"
        # ---------------------------------------------------------------------/



    def _check_if_any_crop_dir_exists(self):
        """
        """
        replace = True #  TODO:  可以在 config 多加一個 replace 參數，選擇要不要重切

        existing_crop_dir = []
        for dir in [self.save_dir_train, self.save_dir_test]:
            temp_list = list(dir.glob(f"*/{self.crop_dir_name}"))
            if temp_list:
                self._cli_out.write(f"{Fore.YELLOW}{Back.BLACK} Detect {len(temp_list)} '{self.crop_dir_name}' directories in '{dir}' {Style.RESET_ALL}")
            existing_crop_dir.extend(temp_list)

        if existing_crop_dir:
            if replace: # (config varname TBA)
                self._cli_out.write(f"Deleting {len(existing_crop_dir)} '{self.crop_dir_name}' directories... ")
                for dir in existing_crop_dir: shutil.rmtree(dir)
                self._cli_out.write(f"{Fore.GREEN}{Back.BLACK} Done! {Style.RESET_ALL}")
            else:
                raise FileExistsError(f"{Fore.YELLOW}{Back.BLACK} To re-crop the images, "
                                      f"set `config.replace` = True {Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/



    def run(self, config_file:Union[str, Path]="1.make_dataset.toml"):
        """
        """
        self._cli_out.divide()
        self._set_attrs(config_file)
        
        self._check_if_any_crop_dir_exists()
        train_img_paths = list(self.save_dir_train.glob(f"*/*.tiff"))
        test_img_paths = list(self.save_dir_test.glob(f"*/*.tiff"))
        img_paths = sorted(test_img_paths + train_img_paths, key=dsname.get_dsname_sortinfo)
        self._cli_out.write(f"found {len(img_paths)} images, "
                            f"train: {len(train_img_paths)}, "
                            f"test: {len(test_img_paths)}")
        
        self._cli_out.divide()
        with tqdm(total=len(img_paths), desc=f"[ {self._cli_out.logger_name} ] : ") as pbar:
            
            for path in img_paths:
                
                img = cv2.imread(str(path))
                
                """ Extract info """
                path_split = str(path).split(os.sep)
                fish_dsname = path_split[-2]
                file_ext = path_split[-1].split(".")[-1]
                
                """ Generate `crop_dir` """
                crop_dir = path_split[:-1]
                crop_dir.append(self.crop_dir_name)
                crop_dir = Path(os.sep.join(crop_dir))
                create_new_dir(crop_dir)
                
                # cropping
                crop_img_list = gen_crop_img(img, self.config)
                
                if pbar.total != len(img_paths)*len(crop_img_list):
                    pbar.total = len(img_paths)*len(crop_img_list)
                
                for i, cropped_img in enumerate(crop_img_list):
                    
                    cropped_name = f"{fish_dsname}_crop_{i:{formatter_padr0(crop_img_list)}}"
                    
                    pbar.desc = f"[ {self._cli_out.logger_name} ] {cropped_name} : "
                    pbar.refresh()
                    
                    save_path = crop_dir.joinpath(f"{cropped_name}.{file_ext}")
                    cv2.imwrite(str(save_path), cropped_img)
                    
                    pbar.update(1)
                    pbar.refresh()
        
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/