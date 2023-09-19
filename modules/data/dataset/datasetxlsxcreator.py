import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

import cv2
import pandas as pd
from tomlkit.toml_document import TOMLDocument
from colorama import Fore, Back, Style
from tqdm.auto import tqdm

from . import dsname
from .utils import gen_dataset_xlsx_name_dict, drop_too_dark
from .. import dname
from ..processeddatainstance import ProcessedDataInstance
from ...shared.clioutput import CLIOutput
from ...shared.config import load_config
from ...shared.pathnavigator import PathNavigator
from ...shared.utils import create_new_dir, get_target_str_idx_in_list
# -----------------------------------------------------------------------------/


class DatasetXLSXCreator():


    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        self._path_navigator = PathNavigator()
        self.processed_data_instance = ProcessedDataInstance()
        self._cli_out = CLIOutput(display_on_CLI, 
                                  logger_name="Dataset XLSX Creator")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        self.config: Union[dict, TOMLDocument] # self._set_attrs()
        
        self.palmskin_result_alias: str # self._set_config_attrs()
        self.cluster_desc: str # self._set_config_attrs()
        self.random_seed: int # self._set_config_attrs()
        self.dynamic_select: bool # self._set_config_attrs()
        
        self.save_dir_root: Path # self._set_save_dirs()
        self.save_dir_train: Path # self._set_save_dirs()
        self.save_dir_test: Path # self._set_save_dirs()
        
        self.crop_dir_name: str # self._set_crop_dir_name()
        
        self.classif_strategy: str # self._set_classif_strategy()
        
        self.dataset_xlsx_dir: Path # self._set_dataset_xlsx_attrs()
        self.dataset_xlsx_path: Path # self._set_dataset_xlsx_attrs()
        # ---------------------------------------------------------------------/



    def _set_attrs(self, config_file:Union[str, Path]):
        """
        """
        self.config: Union[dict, TOMLDocument] = load_config(config_file, cli_out=self._cli_out)
        self.processed_data_instance.set_attrs(config_file)
        self._set_config_attrs()
        self._set_save_dirs()
        self._set_crop_dir_name()
        self._set_classif_strategy()
        self._set_dataset_xlsx_attrs()
        # ---------------------------------------------------------------------/



    def _set_config_attrs(self):
        """ Set below attributes
            - `self.palmskin_result_alias`: str
            - `self.cluster_desc`: str
            - `self.random_seed`: int
            - `self.dynamic_select`: bool
        """
        """ [data_processed] """
        self.palmskin_result_alias: str = self.config["data_processed"]["palmskin_result_alias"]
        self.cluster_desc: str = self.config["data_processed"]["cluster_desc"]
        
        """ [param] """
        self.random_seed: int = self.config["param"]["random_seed"]
        self.dynamic_select: bool = self.config["param"]["dynamic_select"]
        # ---------------------------------------------------------------------/



    def _set_save_dirs(self):
        """ Set below attributes
            - `self.save_dir_root`: Path
            - `self.save_dir_train`: Path
            - `self.save_dir_test`: Path
        """
        dataset_cropped: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v2")
        
        self.save_dir_root: Path = dataset_cropped.joinpath(f"SEED_{self.random_seed}",
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
        temp_dict = gen_dataset_xlsx_name_dict(self.config)
        
        self.crop_dir_name: str = f"{temp_dict['crop_size']}_{temp_dict['shift_region']}"
        # ---------------------------------------------------------------------/



    def _set_classif_strategy(self):
        """
        """
        cluster_desc_split = self.cluster_desc.split("_") # ['SURF3C', 'KMeansORIG', 'RND2022']
        cluster_desc_split.pop(0) # ['KMeansORIG', 'RND2022']
        self.classif_strategy: str = "_".join(cluster_desc_split) # 'KMeansORIG_RND2022'
        # ---------------------------------------------------------------------/



    def _set_dataset_xlsx_attrs(self):
        """ Set below attributes
            - `self.dataset_xlsx_dir`: Path
            - `self.dataset_xlsx_path`: Path
        """
        self.dataset_xlsx_dir: Path = self.save_dir_root.joinpath(self.classif_strategy)
        
        name_dict = gen_dataset_xlsx_name_dict(self.config)
        dataset_xlsx_name = "_".join(name_dict.values())
        
        self.dataset_xlsx_path: Path = self.dataset_xlsx_dir.joinpath(f"{dataset_xlsx_name}.xlsx")
        if self.dataset_xlsx_path.exists():
            raise FileExistsError(f"{Fore.RED}{Back.BLACK} target `dataset_xlsx` already exists: "
                                  f"'{self.dataset_xlsx_path}' {Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/



    def run(self, config_file:Union[str, Path]="1.make_dataset.toml"):
        """
        """
        self._cli_out.divide()
        self._set_attrs(config_file)
        
        """ Load `clustered_xlsx`, add a fish_id column to retrieve the class """
        clustered_xlsx_df: pd.DataFrame = pd.read_excel(self.processed_data_instance.clustered_xlsx_files_dict[
                                                             self.cluster_desc], engine='openpyxl')
        clustered_xlsx_df['fish_id'] = clustered_xlsx_df["Brightfield"].apply(lambda x: dname.get_dname_sortinfo(x)[0])
        create_new_dir(self.dataset_xlsx_dir)
        
        """ Get images """
        if self.dynamic_select: train_img_paths = list(self.save_dir_train.glob(f"*/*.tiff"))
        else: train_img_paths = list(self.save_dir_train.glob(f"*/{self.crop_dir_name}/*.tiff"))
        test_img_paths = list(self.save_dir_test.glob(f"*/{self.crop_dir_name}/*.tiff"))
        img_paths = sorted(test_img_paths + train_img_paths, key=dsname.get_dsname_sortinfo)
        self._cli_out.write(f"found {len(img_paths)} images, "
                            f"train: {len(train_img_paths)}, "
                            f"test: {len(test_img_paths)}")
        
        dataset_df: Union[None, pd.DataFrame] = None
        self._cli_out.divide()
        with tqdm(total=len(img_paths), desc=f"[ {self._cli_out.logger_name} ] : ") as pbar:
            
            for path in img_paths:
                
                path_split = str(path).split(os.sep)
                
                """ Get `dsname` """
                fish_dsname = path_split[-1].split(".")[0]
                pbar.desc = f"[ {self._cli_out.logger_name} ] {fish_dsname} : "
                pbar.refresh()
                
                """ Get `parent_dsname` ( without `_crop_[idx]` ) """
                target_idx = get_target_str_idx_in_list(path_split, self.palmskin_result_alias)
                parent_dsname = path_split[target_idx+2]
                parent_dsname_split = parent_dsname.split("_")
                fish_id     = int(parent_dsname_split[1])
                fish_pos    = parent_dsname_split[2]
                cut_section = parent_dsname_split[3]
                
                """ Get `fish_class` """
                df_filtered_rows = clustered_xlsx_df[clustered_xlsx_df['fish_id'] == fish_id]
                df_filtered_rows = df_filtered_rows.reset_index(drop=True)
                fish_class = df_filtered_rows.loc[0, "class"]
                
                """ preserve / discard """
                if self.dynamic_select:
                    state = "preserve"
                    darkratio = "---"
                else:
                    img = cv2.imread(str(path))
                    
                    select, drop = drop_too_dark([img], self.config)
                    if select is None: assert drop, "Either `select` or `drop` needs to be empty"
                    if drop is None: assert select, "Either `select` or `drop` needs to be empty"
                    
                    if select:
                        state = "preserve"
                        darkratio = select[0][2]
                        
                    if drop:
                        state = "discard"
                        darkratio = drop[0][2]
                
                """ Create `temp_dict` """
                temp_dict = {}
                # -------------------------------------------------------
                temp_dict["image_name"] = fish_dsname
                temp_dict["class"] = fish_class
                # -------------------------------------------------------
                temp_dict["parent (dsname)"] = parent_dsname
                temp_dict["fish_id"] = fish_id
                temp_dict["fish_pos"] = fish_pos
                temp_dict["cut_section"] = cut_section
                # -------------------------------------------------------
                temp_dict["dataset"] = path_split[target_idx+1]
                temp_dict["darkratio"] = darkratio
                temp_dict["state"] = state
                # -------------------------------------------------------
                temp_dict["path"] = path
                # -------------------------------------------------------
                temp_df = pd.DataFrame(temp_dict, index=[0])
                
                # add to `dataset_df`
                if dataset_df is None: dataset_df = temp_df.copy()
                else: dataset_df = pd.concat([dataset_df, temp_df], ignore_index=True)
                
                pbar.update(1)
                pbar.refresh()
        
        """ Save `dataset_xlsx` """
        self._cli_out.new_line()
        self._cli_out.write("Saving `dataset_xlsx`... ")
        dataset_df.to_excel(self.dataset_xlsx_path, engine="openpyxl")
        self._cli_out.write(f"{Fore.GREEN}{Back.BLACK} Done! {Style.RESET_ALL}")
        # ---------------------------------------------------------------------/