import os
import sys
import re
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Union
from colorama import Fore, Back, Style
from collections import OrderedDict
import json
import toml
from logging import Logger
from tomlkit.toml_document import TOMLDocument

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import cv2

from . import dname
from .collect_data import resave_result
from ..shared.logger import init_logger
from ..shared.pathnavigator import PathNavigator
from ..shared.utils import create_new_dir, load_config
from ..assert_fn import *


class ProcessedDataInstance():
    
    def __init__(self) -> None:
        """
        """
        self._logger = init_logger(r"Processed Data Instance")
        self._display_kwargs = {
            "display_on_CLI": True,
            "logger": self._logger
        }
        self._path_navigator = PathNavigator()
        
        # -----------------------------------------------------------------------------------
        self.data_processed_root:Union[None, Path] = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("data_processed", **self._display_kwargs)
        
        # -----------------------------------------------------------------------------------
        # variables
        
        self.instance_root:Union[None, Path] = None
        self.instance_desc:Union[None, str] = None
        self.instance_name:Union[None, str] = None
        
        self.palmskin_processed_dir:Union[None, Path] = None
        self.palmskin_processed_reminder:Union[None, str] = None
        self.palmskin_processed_dname_dirs_dict:Dict[str, Path] = {}
        self.palmskin_processed_config:dict = {}
        self.palmskin_processed_alias_map:Dict[str, str] = {}
        
        self.brightfield_processed_dir:Union[None, Path] = None
        self.brightfield_processed_reminder:Union[None, str] = None
        self.brightfield_processed_dname_dirs_dict:Dict[str, Path] = {}
        self.brightfield_processed_config:dict = {}
        self.brightfield_processed_alias_map:Dict[str, str] = {}
        
        self.palmskin_recollect_dir:Union[None, Path] = None
        self.palmskin_recollected_dirs_dict:Dict[str, Path] = {}
        
        self.brightfield_recollect_dir:Union[None, Path] = None
        self.brightfield_recollected_dirs_dict:Dict[str, Path] = {}
        
        self.data_xlsx_path:Union[None, Path] = None
        
        self.clustered_xlsx_dir:Union[None, Path] = None
        self.clustered_xlsx_files_dict:Dict[str, Path] = {}
        
        # End section -----------------------------------------------------------------------------------
    
    
    
    def load_config(self, config_name:str):
        """
        """
        self.config = load_config(config_name, **self._display_kwargs)
        self._set_instance_root()
        self._set_processed_dirs()
        self._set_processed_dname_dirs_dicts()
        self._set_processed_configs()
        self._set_processed_alias_maps()
        self._set_recollect_dirs()
        self._set_data_xlsx_path()
        self._set_clustered_xlsx_dir()
    
    
    
    def _set_instance_root(self):
        """ Set below attributes
            1. `self.instance_root`
            2. `self.instance_desc`
            3. `self.instance_name`
        """
        path = self._path_navigator.processed_data.get_instance_root(self.config, **self._display_kwargs)
        assert_dir_exists(path)
        
        self.instance_root = path
        self.instance_desc = self.config["data_processed"]["instance_desc"]
        self.instance_name = str(self.instance_root).split(os.sep)[-1]
    
    
    
    def _set_processed_dirs(self):
        """ Set below attributes
            1. `self.palmskin_processed_dir`
            2. `self.palmskin_processed_reminder`
            3. `self.brightfield_processed_dir`
            4. `self.brightfield_processed_reminder`
        """
        """ palmskin """
        path = self._path_navigator.processed_data.get_processed_dir("palmskin", self.config, **self._display_kwargs)
        if path:
            self.palmskin_processed_dir = path
            self.palmskin_processed_reminder = re.split("{|}", str(path).split(os.sep)[-1])[1]
        else:
            raise FileNotFoundError("Can't find any 'PalmSkin_preprocess' directory, "
                                    "please run `0.2.preprocess_palmskin.py` to preprocess your palmskin images first.")
        
        """ brightfield """
        path = self._path_navigator.processed_data.get_processed_dir("brightfield", self.config, **self._display_kwargs)
        if path:
            self.brightfield_processed_dir = path
            self.brightfield_processed_reminder = re.split("{|}", str(path).split(os.sep)[-1])[1]
        else:
            raise FileNotFoundError("Can't find any 'BrightField_analyze' directory, "
                                    "please run `0.3.analyze_brightfield.py` to analyze your brightfield images first.")
    
    
    
    def _scan_processed_dname_dirs(self, image_type:str):
        """

        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only")
        
        processed_dir:Path = getattr(self, f"{image_type}_processed_dir")
        dname_dirs = processed_dir.glob("*")
        
        dname_dirs_dict = {str(dname_dir).split(os.sep)[-1]: dname_dir
                                                         for dname_dir in dname_dirs}
        
        for key in list(dname_dirs_dict.keys()):
            if key == "!~delete": dname_dirs_dict.pop(key) # rm "!~delete" directory
            if ".log" in key: dname_dirs_dict.pop(key) # rm "log" files
            if ".toml" in key: dname_dirs_dict.pop(key) # rm "toml" file
        
        dname_dirs_dict = OrderedDict(sorted(list(dname_dirs_dict.items()), key=lambda x: dname.get_dname_sortinfo(x[0])))
        
        return dname_dirs_dict
    
    
    
    def _update_instance_postfixnum(self):
        """
        """
        old_name = self.instance_name
        new_name = f"{{{self.instance_desc}}}_Academia_Sinica_i{len(self.palmskin_processed_dname_dirs_dict)}"
        
        if new_name != old_name:
            self._logger.info("Update Instance Postfix Number ...")
            os.rename(self.instance_root, self.data_processed_root.joinpath(new_name))
            self._set_instance_root()
            self._set_processed_dirs()
    
    
    
    def _set_processed_dname_dirs_dicts(self):
        """ Set below attributes, run functions : 
            1. `self.palmskin_processed_dname_dirs_dict`
            2. `self._update_instance_postfixnum()`
            3. `self.brightfield_processed_dname_dirs_dict`
        """
        """ palmskin """
        self.palmskin_processed_dname_dirs_dict = self._scan_processed_dname_dirs("palmskin")
        self._update_instance_postfixnum()
        
        """ brightfield """
        self.brightfield_processed_dname_dirs_dict = self._scan_processed_dname_dirs("brightfield")
    
    
    
    def _set_recollect_dirs(self):
        """ Set below attributes 
            1. `self.palmskin_recollect_dir`
            2. `self.palmskin_recollected_dirs_dict`
            3. `self.brightfield_recollect_dir`
            4. `self.brightfield_recollected_dirs_dict`
        """
        """ palmskin """
        path = self._path_navigator.processed_data.get_recollect_dir("palmskin", self.config, **self._display_kwargs)
        if self.palmskin_recollect_dir != path:
            self.palmskin_recollect_dir = path
            self._update_recollected_dirs_dict("palmskin")
        else:
            self._update_recollected_dirs_dict("palmskin")
        
        """ brightfield """
        path = self._path_navigator.processed_data.get_recollect_dir("brightfield", self.config, **self._display_kwargs)
        if self.brightfield_recollect_dir != path:
            self.brightfield_recollect_dir = path
            self._update_recollected_dirs_dict("brightfield")
        else:
            self._update_recollected_dirs_dict("brightfield")
    
    
    
    def _update_recollected_dirs_dict(self, image_type:str):
        """

        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only")
        
        setattr(self, f"{image_type}_recollected_dirs_dict", {}) # reset variable
        recollected_dict = getattr(self, f"{image_type}_recollected_dirs_dict")
        
        recollect_dir:Union[None, Path] = getattr(self, f"{image_type}_recollect_dir")
        if recollect_dir is not None:
            """ Scan directories """
            found_list = sorted(list(recollect_dir.glob("*")), key=lambda x: str(x))
            for recollected_dir in found_list:
                recollected_name = str(recollected_dir).split(os.sep)[-1]
                recollected_dict[recollected_name] = recollected_dir
    
    
    
    def _set_data_xlsx_path(self):
        """ Set below attributes
            1. `self.data_xlsx_path`
        """
        self.data_xlsx_path = self._path_navigator.processed_data.get_data_xlsx_path(self.config, **self._display_kwargs)
    
    
    
    def _set_clustered_xlsx_dir(self):
        """ Set below attributes
            1. `self.clustered_xlsx_dir`
            2. `self.clustered_xlsx_files_dict`
        """
        path = self._path_navigator.processed_data.get_clustered_xlsx_dir(self.config, **self._display_kwargs)
        if self.clustered_xlsx_dir != path:
            self.clustered_xlsx_dir = path
            self._update_clustered_xlsx_files_dict()
        else:
            self._update_clustered_xlsx_files_dict()
    
    
    
    def _update_clustered_xlsx_files_dict(self):
        """
        """
        self.clustered_xlsx_files_dict = {} # reset variable
        
        if self.clustered_xlsx_dir is not None:
            """ Scan files """
            found_list = sorted(list(self.clustered_xlsx_dir.glob(r"{*}_data.xlsx")), key=lambda x: str(x))
            for xlsx_file in found_list:
                xlsx_name = str(xlsx_file).split(os.sep)[-1]
                cluster_desc = re.split("{|}", xlsx_name)[1]
                self.clustered_xlsx_files_dict[cluster_desc] = xlsx_file
    
    
    
    def _load_processed_config(self, image_type:str):
        """

        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only")
        
        if image_type == "palmskin":
            target_text = "palmskin_preprocess"
        elif image_type == "brightfield":
            target_text = "brightfield_analyze"
        
        processed_dir:Path = getattr(self, f"{image_type}_processed_dir")
        config_path = processed_dir.joinpath(f"{target_text}_config.toml")
        assert_file_exists(config_path)
        
        with open(config_path, mode="r") as f_reader:
            config = toml.load(f_reader)
        
        setattr(self, f"{image_type}_processed_config", config)
    
    
    
    def _load_processed_alias_map(self, image_type:str):
        """

        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only")
        
        processed_dir:Path = getattr(self, f"{image_type}_processed_dir")
        map_path = processed_dir.joinpath(f"{image_type}_result_alias_map.toml")
        assert_file_exists(map_path)
        
        with open(map_path, mode="r") as f_reader:
            alias_map = toml.load(f_reader)
        
        setattr(self, f"{image_type}_processed_alias_map", alias_map)
    
    
    
    def _set_processed_configs(self):
        """ Set below attributes
            1. `self.palmskin_processed_config`
            2. `self.brightfield_processed_config`
        """
        self._load_processed_config("palmskin")
        self._load_processed_config("brightfield")
    
    
    
    def _set_processed_alias_maps(self):
        """ Set below attributes
            1. `self.palmskin_processed_alias_map`
            2. `self.brightfield_processed_alias_map`
        """
        self._load_processed_alias_map("palmskin")
        self._load_processed_alias_map("brightfield")
    
    
    
    # def _init_brightfield_analyze_alias_map(self):
        
    #     with open(self.brightfield_processed_dir.joinpath("brightfield_analyze_config.toml"), mode="r") as f_reader:
    #         self.brightfield_processed_config = toml.load(f_reader)
        
    #     analyze_kwargs = self.brightfield_processed_config["param"]
    #     autothreshold_algo = analyze_kwargs['auto_threshold']
        
    #     self.brightfield_processed_alias_map = {
    #         "original_16bit" :          "MetaImage/*_original_16bit.tif",
    #         "cropped_BF" :              "*_cropped_BF.tif", # CHECK_PT 
    #         "AutoThreshold" :           f"MetaImage/*_AutoThreshold_{autothreshold_algo}.tif",
    #         "measured_mask" :           "MetaImage/*_measured_mask.tif",
    #         "cropped_BF--MIX" :         "*_cropped_BF--MIX.tif", # CHECK_PT 
    #         "RoiSet" :                  "MetaImage/RoiSet.zip",
    #         "AutoAnalysis" :            "AutoAnalysis.csv",
    #         "ManualAnalysis" :          "ManualAnalysis.csv",
    #         "Manual_measured_mask" :    "Manual_measured_mask.tif", # CHECK_PT 
    #         "Manual_cropped_BF--MIX" :  "Manual_cropped_BF--MIX.tif", # CHECK_PT 
    #     }
    
    
    
    # def get_existing_processed_results(self, processed_name:str, result_alias:str) -> Tuple[str, List[Path]]:
    #     """

    #     Args:
    #         processed_name (str): `'BrightField_analyze'` or `'PalmSkin_preprocess'`
    #         result_alias (str): please refer to `'Documents/{NamingRule}_ResultAlias.md'` in this repository

    #     Returns:
    #         Tuple[str, List[Path]]: `(relative_path_in_fish_dir, sorted_results)`
    #     """
    #     assert (processed_name == "BrightField_analyze") or (processed_name == "PalmSkin_preprocess"), \
    #         f"processed_name = '{processed_name}', accept 'BrightField_analyze' or 'PalmSkin_preprocess' only"
    #     processed_name_lower = processed_name.lower()
        
    #     processed_dir:Path = getattr(self, f"{processed_name_lower}_dir")
    #     alias_map = getattr(self, f"{processed_name_lower}_alias_map")
        
    #     assert alias_map[result_alias]
    #     rel_path_in_fish_dir = alias_map[result_alias]
        
    #     # regex filter
    #     results = sorted(processed_dir.glob(f"*/{alias_map[result_alias]}"), key=get_fish_id_pos)
    #     pattern = rel_path_in_fish_dir.split("/")[-1]
    #     pattern = pattern.replace("*", r"[0-9]*")
    #     num = 0
    #     actual_name = None
    #     for _ in range(len(results)):
    #         result_name = str(results[num]).split(os.sep)[-1]
    #         if not re.fullmatch(pattern, result_name):
    #             results.pop(num)
    #         else:
    #             num += 1
    #             if actual_name is None: actual_name = result_name
        
    #     if rel_path_in_fish_dir.split("/")[0] == "MetaImage": return f"MetaImage/{actual_name}", results
    #     else: return actual_name, results
    
    
    
    # def collect_results(self, processed_name:str, result_alias:str, log_mode:str="missing"):
    #     """

    #     Args:
    #         processed_name (str): `'BrightField_analyze'` or `'PalmSkin_preprocess'`
    #         result_alias (str): please refer to `'Documents/{NamingRule}_ResultAlias.md'` in this repository
    #         log_mode (str, optional): `'missing'` or `'finding'`. Defaults to "missing".
    #     """
    #     assert (processed_name == "BrightField_analyze") or (processed_name == "PalmSkin_preprocess"), \
    #         f"processed_name = '{processed_name}', accept 'BrightField_analyze' or 'PalmSkin_preprocess' only"
            
    #     assert (log_mode == "missing") or (log_mode == "finding"), \
    #         f"log_mode = '{log_mode}', accept 'missing' or 'finding' only"
        
    #     processed_name_lower = processed_name.lower()
    #     source_name = processed_name.split("_")[0]
    #     source_name_lower = source_name.lower()
        
    #     # get attributes
    #     processed_reminder = getattr(self, f"{processed_name_lower}_reminder")
    #     alias_map = getattr(self, f"{processed_name_lower}_alias_map")
    #     recollect_dir:Path = getattr(self, f"{source_name_lower}_recollect_dir")
    #     assert alias_map[result_alias]
        
    #     # output
    #     output_dir = self.instance_root.joinpath(f"{{{processed_reminder}}}_{source_name}_reCollection", result_alias)
    #     assert not output_dir.exists(), f"Directory: '{output_dir}' already exists, please delete it before collecting results."
    #     create_new_dir(output_dir)
        
    #     relative_path_in_fish_dir, results = self.get_existing_processed_results(processed_name, result_alias)
        
    #     summary = {}
    #     summary["result_alias"] = result_alias
    #     summary["actual_name"] = relative_path_in_fish_dir.split("/")[-1]
    #     summary["max_probable_num"] = get_fish_id_pos(results[-1])[0]
    #     summary["total files"] = len(results)
    #     summary[log_mode] = []
        
    #     previous_fish = ""
    #     for i in range(summary["max_probable_num"]):
            
    #         one_base_iter_num = i+1
            
    #         if source_name == "PalmSkin": pos_list = ["A", "P"]
    #         else: pos_list = [""]
            
            
    #         for pos in pos_list:
                
    #             # expect_name
    #             if source_name == "PalmSkin": expect_name = f"{one_base_iter_num}_{pos}"
    #             else: expect_name = f"{one_base_iter_num}" # BrightField
                
    #             try: # current_name
                    
    #                 fish_ID, fish_pos = get_fish_id_pos(results[0])
    #                 if source_name == "PalmSkin": current_name = f"{fish_ID}_{fish_pos}"
    #                 else: current_name = f"{fish_ID}" # BrightField
    #                 assert current_name != previous_fish, f"fish_dir repeated!, check '{previous_fish}' "
                
    #             except: pass
                
    #             # comparing
    #             if current_name == expect_name:
                    
    #                 path = results.pop(0)
    #                 resave_result(path, output_dir, alias_map[result_alias])
    #                 previous_fish = current_name
    #                 if log_mode == "finding": summary[log_mode].append(f"{expect_name}")
                    
    #             else: 
    #                 if log_mode == "missing": summary[log_mode].append(f"{expect_name}")

        
    #     summary[f"len({log_mode})"] = len(summary[log_mode])
    #     print(json.dumps(summary, indent=4))
        
    #     # write log
    #     log_path = output_dir.joinpath(f"{{Logs}}_collect_{processed_name_lower}_results.log")
    #     with open(log_path, mode="w") as f_writer:
    #         json.dump(summary, f_writer, indent=4)
        
    #     # update `recollect_dir`
    #     if recollect_dir is None: self._check_recollect_dir(source_name)
    #     else: self._update_recollected_dirs_dict(source_name)
    
    
    
    # def create_data_xlsx(self, logger:Logger):
    #     """To generate data information in XLSX ( XLSX file will used to compute the classes in classification task ):

    #         All fish will process with the following step : 
            
    #             1. Run ImageJ Macro : Use bright field (BF) images to compute the surface area (SA) and surface length (SL), and store their results in CSV format.
    #             2. Collect all generated CSV files using pandas.DataFrame().
    #             3. Use `fish_id` to find and add their `palmskin_RGB` images into the DataFrame.
    #             4. Save results in XLSX format.

    #     Args:
    #         logger (Logger): external logger created using package `logging`
    #     """
        
    #     # -----------------------------------------------------------------------------------
    #     # BrightField
        
    #     # Scan `AutoAnalysis` results, and sort ( Due to OS scanning strategy 10 may listed before 8 )
    #     bf_recollect_auto_list = self.get_existing_processed_results("BrightField_analyze", "AutoAnalysis")[1]
    #     bf_recollect_auto_list = sorted(bf_recollect_auto_list, key=get_fish_id_pos)

    #     # Scan `ManualAnalysis` results, and sort ( Due to OS scanning strategy 10 may listed before 8 )
    #     bf_recollect_manual_list = self.get_existing_processed_results("BrightField_analyze", "ManualAnalysis")[1]
    #     bf_recollect_manual_list = sorted(bf_recollect_manual_list, key=get_fish_id_pos)

    #     # show info
    #     logger.info((f"BrightField: Found {len(bf_recollect_auto_list)} AutoAnalysis.csv, "
    #             f"{len(bf_recollect_manual_list)} ManualAnalysis.csv, "
    #             f"Total: {len(bf_recollect_auto_list) + len(bf_recollect_manual_list)} files"))

    #     # Merge `AutoAnalysis` and `ManualAnalysis` list
    #     bf_recollect_auto_dict = create_dict_by_fishid(bf_recollect_auto_list)
    #     bf_recollect_manual_dict = create_dict_by_fishid(bf_recollect_manual_list)
    #     bf_recollect_merge_dict = merge_bf_analysis(bf_recollect_auto_dict, bf_recollect_manual_dict)
    #     bf_recollect_merge_list = sorted(list(bf_recollect_merge_dict.values()), key=get_fish_id_pos)
    #     logger.info(f"--> After Merging , Total: {len(bf_recollect_merge_list)} files")
        
    #     # -----------------------------------------------------------------------------------
    #     # PalmSkin

    #     palmskin_preprocess_fish_dirs = list(self.palmskin_processed_dname_dirs_dict.keys())
    #     logger.info(f"PalmSkin: Found {len(palmskin_preprocess_fish_dirs)} tif files")
        
    #     # -----------------------------------------------------------------------------------
    #     # Processing

    #     delete_uncomplete_row = True
    #     output = os.path.join(self.instance_root, r"data.xlsx")

    #     # Creating "data.xlsx"
    #     data = pd.DataFrame(columns=["BrightField name with Analysis statement (CSV)",
    #                                 "Anterior (SP8, .tif)", 
    #                                 "Posterior (SP8, .tif)",
    #                                 "Trunk surface area, SA (um2)",
    #                                 "Standard Length, SL (um)"])


    #     print("\n\nprocessing...\n")

    #     # Variable
    #     max_probable_num = get_fish_id_pos(bf_recollect_merge_list[-1])[0]
    #     logger.info(f'max_probable_num {type(max_probable_num)}: {max_probable_num}\n')


    #     # Starting...
    #     for i in range(max_probable_num):
            
    #         # *** Print CMD section divider ***
    #         print("="*100, "\n")
            
    #         one_base_iter_num = i+1 # Make iteration starting number start from 1
    #         logger.info(f'one_base_iter_num {type(one_base_iter_num)}: {one_base_iter_num}\n')
            
            
    #         if  one_base_iter_num == get_fish_id_pos(bf_recollect_merge_list[0])[0] :
                
    #             # Get info strings
    #             bf_result_path = bf_recollect_merge_list.pop(0)
    #             bf_result_path_split = str(bf_result_path).split(os.sep)
    #             bf_result_name = bf_result_path_split[-2] # `AutoAnalysis` or `ManualAnalysis`
    #             bf_result_analysis_type = bf_result_path_split[-1].split(".")[0] # Get name_noExtension
    #             logger.info(f'bf_result_name {type(bf_result_name)}: {bf_result_name}')
    #             logger.info(f'analysis_type {type(bf_result_analysis_type)}: {bf_result_analysis_type}')
    #             # Read CSV
    #             analysis_csv = pd.read_csv(bf_result_path, index_col=" ")
    #             assert len(analysis_csv) == 1, f"More than 1 measure data in csv file, file:{bf_result_path}"
    #             # Get surface area from analysis file
    #             surface_area = analysis_csv.loc[1, "Area"]
    #             logger.info(f'surface_area {type(surface_area)}: {surface_area}')
    #             # Get standard length from analysis file
    #             standard_length = analysis_csv.loc[1, "Feret"]
    #             logger.info(f'standard_length {type(standard_length)}: {standard_length}')
                
    #             data.loc[one_base_iter_num, "BrightField name with Analysis statement (CSV)"] = f"{bf_result_name}_{bf_result_analysis_type}"
    #             data.loc[one_base_iter_num, "Trunk surface area, SA (um2)"] = surface_area
    #             data.loc[one_base_iter_num, "Standard Length, SL (um)"] = standard_length

    #         else: data.loc[one_base_iter_num] = np.nan # Can't find corresponding analysis result, make an empty row.
            
            
    #         if f"{one_base_iter_num}_A" in palmskin_preprocess_fish_dirs[0]:
    #             palmskin_RGB_A_name = palmskin_preprocess_fish_dirs.pop(0)
    #             logger.info(f'palmskin_RGB_A_name {type(palmskin_RGB_A_name)}: {palmskin_RGB_A_name}')
    #             data.loc[one_base_iter_num, "Anterior (SP8, .tif)" ] =  palmskin_RGB_A_name
            
            
    #         if f"{one_base_iter_num}_P" in palmskin_preprocess_fish_dirs[0]:
    #             palmskin_RGB_P_name = palmskin_preprocess_fish_dirs.pop(0)
    #             logger.info(f'palmskin_RGB_P_name {type(palmskin_RGB_P_name)}: {palmskin_RGB_P_name}')
    #             data.loc[one_base_iter_num, "Posterior (SP8, .tif)" ] =  palmskin_RGB_P_name
            
            
    #         print("\n\n\n")


    #     if delete_uncomplete_row: data.dropna(inplace=True)
    #     data.to_excel(output, engine="openpyxl")

    #     self._check_data_xlsx_path()
    
    
    
    def __repr__(self):
        
        repr_string = "{\n"
        repr_string += f"self.data_processed_root {type(self.data_processed_root)}: '{self.data_processed_root}'\n\n"
                
        repr_string += f"self.instance_desc : '{self.instance_desc}'\n"
        repr_string += f"self.instance_root : '{self.instance_root}'\n"
        repr_string += f"self.instance_name : '{self.instance_name}'\n\n"
                
        repr_string += f"self.palmskin_preprocess_dir : '{self.palmskin_processed_dir}'\n"
        repr_string += f"self.palmskin_preprocess_reminder : '{self.palmskin_processed_reminder}'\n\n"
        # repr_string += f"self.palmskin_preprocess_config.param : {json.dumps(self.palmskin_processed_config["param"], indent=4)}\n\n"
                
        repr_string += f"self.brightfield_analyze_dir : '{self.brightfield_processed_dir}'\n"
        repr_string += f"self.brightfield_analyze_reminder : '{self.brightfield_processed_reminder}'\n\n"
        # repr_string += f"self.brightfield_analyze_config.param : {json.dumps(self.brightfield_processed_config["param"], indent=4)}\n\n"
                
        repr_string += f"self.palmskin_recollect_dir : '{self.palmskin_recollect_dir}'\n"
        # repr_string += f"self.palmskin_recollected_dirs_dict : {json.dumps(list(self.palmskin_recollected_dirs_dict.keys()), indent=4)}\n\n"
                
        repr_string += f"self.brightfield_recollect_dir : '{self.brightfield_recollect_dir}'\n\n"
        # repr_string += f"self.brightfield_recollected_dirs_dict : {json.dumps(list(self.brightfield_recollected_dirs_dict.keys()), indent=4)}\n\n"
                
        repr_string += f"self.data_xlsx_path : '{self.data_xlsx_path}'\n"
                
        repr_string += f"self.clustered_xlsx_dir : '{self.clustered_xlsx_dir}'\n"
        # repr_string += f"self.clustered_xlsx_paths_dict : {json.dumps(list(self.clustered_xlsx_paths_dict.keys()), indent=4)}\n\n"
        repr_string += "}\n"
        
        return repr_string
    
    
    
    def check_palmskin_images_condition(self, palmskin_result_alias:str, xlsx_name:str=None) -> str:
        """Check the existence and readability of the palm skin images recorded in the XLSX file.

        Args:
            palmskin_result_alias (str): please refer to `'Documents/{NamingRule}_ResultAlias.md'` in this repository.
            xlsx_name (str, optional): If `None`, use `self.data_xlsx_path`

        Returns:
            Tuple[bool, Union[str, None]]: if the check is passed, return the `relative_path_in_fish_dir` of `self.get_existing_processed_results()`.
        """
        if xlsx_name is None:
            assert self.data_xlsx_path is not None, \
                f"{Fore.RED}{Back.BLACK} Can't find `data.xlsx` please use `self.create_data_xlsx()` to create it. {Style.RESET_ALL}\n"
            xlsx_path = self.data_xlsx_path
        
        #  TODO:  xlsx_name is not None, use given xlsx under `Modified_xlsx/`
        
        df_xlsx :pd.DataFrame = pd.read_excel(xlsx_path, engine = 'openpyxl')
        
        palmskin_dnames = list(pd.concat([df_xlsx["Anterior (SP8, .tif)"], df_xlsx["Posterior (SP8, .tif)"]]))
        relative_path_in_fish_dir, processed_palmskin_results = self.get_existing_processed_results("PalmSkin_preprocess", palmskin_result_alias)
        actual_name = relative_path_in_fish_dir.split("/")[-1] 
        
        if relative_path_in_fish_dir.split("/")[0] == "MetaImage": target_idx = -3
        else: target_idx = -2
        processed_palmskin_results = {str(result_path).split(os.sep)[target_idx]: result_path for result_path in processed_palmskin_results}
        
        pbar = tqdm(total=len(palmskin_dnames), desc="Check Image Condition: ")
        read_failed = 0
        for dname in palmskin_dnames:
            pbar.desc = f"Check Image Condition ( {dname} ) : "
            pbar.refresh()
            try:
                path = processed_palmskin_results.pop(dname)
                if cv2.imread(str(path)) is None: 
                    read_failed += 1
                    tqdm.write(f"{Fore.RED}{Back.BLACK}Can't read '{actual_name}' of '{dname}'{Style.RESET_ALL}")
            except:
                tqdm.write(f"{Fore.RED}{Back.BLACK}Can't find '{actual_name}' of '{dname}'{Style.RESET_ALL}")
                read_failed += 1
            pbar.update(1)
            pbar.refresh()
        pbar.close()
        
        if read_failed == 0: print(f"Check Image Condition: {Fore.GREEN}Passed{Style.RESET_ALL}\n")
        else: raise RuntimeError(f"{Fore.RED} Due to broken/non-existing images, the process has been halted. {Style.RESET_ALL}\n")
        
        return relative_path_in_fish_dir