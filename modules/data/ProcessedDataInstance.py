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
        
        setattr(self, f"{image_type}_recollected_dirs_dict", {}) # reset dict
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
    
    
    
    def get_sorted_results(self, image_type:str, result_alias:str) -> Tuple[str, List[Path]]:
        """

        Args:
            image_type (str): `palmskin` or `brightfield`
            result_alias (str): please refer to `result_alias_map.toml` \
                                under `PalmSkin_preprocess` or `BrightField_analyze` directory

        Returns:
            Tuple[str, List[Path]]: `(relative_path_unfder_dname_dir, sorted_results)`
        """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only")
        
        processed_dir:Path = getattr(self, f"{image_type}_processed_dir")
        alias_map = getattr(self, f"{image_type}_processed_alias_map")
        
        assert alias_map[result_alias]
        rel_path:str = alias_map[result_alias]
        sorted_results = sorted(processed_dir.glob(f"*/{rel_path}"), key=dname.get_dname_sortinfo)
        
        return rel_path, sorted_results
    
    
    
    def collect_results(self, image_type:str, result_alias:str, log_mode:str="missing"):
        """ 

        Args:
            image_type (str): `palmskin` or `brightfield`
            result_alias (str): please refer to `result_alias_map.toml` \
                                under `PalmSkin_preprocess` or `BrightField_analyze` directory
            log_mode (str, optional): `missing` or `finding`. Defaults to "missing".
        """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only")
            
        if log_mode not in ["missing", "finding"]:
            raise ValueError(f"log_mode = '{log_mode}', accept 'missing' or 'finding' only")
        
        """ Scan results """
        rel_path, results = self.get_sorted_results(image_type , result_alias)
        
        """ Get `recollect_dir` """
        if image_type == "palmskin": target_text = "PalmSkin"
        elif image_type == "brightfield": target_text = "BrightField"
        reminder = getattr(self, f"{image_type}_processed_reminder")
        recollect_dir = self.instance_root.joinpath(f"{{{reminder}}}_{target_text}_reCollection", result_alias)
        if recollect_dir.exists():
            raise FileExistsError(f"Directory: '{recollect_dir.resolve()}' already exists, please delete it before collecting results.")
        else:
            create_new_dir(recollect_dir)
        
        """ Main process """
        summary = {}
        summary["result_alias"] = result_alias
        summary["file_name"] = rel_path.split(os.sep)[-1]
        summary["max_probable_num"] = dname.get_dname_sortinfo(results[-1])[0]
        summary["total files"] = len(results)
        summary[log_mode] = []
        
        previous_name = ""
        for i in range(summary["max_probable_num"]):
            
            one_base_iter_num = i+1
            
            """ pos option """
            if image_type == "palmskin": pos_list = ["A", "P"]
            else: pos_list = [""] # brightfield
            
            for pos in pos_list:
                
                """ expect_name """
                if image_type == "palmskin": expect_name = f"{one_base_iter_num}_{pos}"
                else: expect_name = f"{one_base_iter_num}" # brightfield
                
                """ current_name """
                fish_id, fish_pos = dname.get_dname_sortinfo(results[0])
                if image_type == "palmskin": current_name = f"{fish_id}_{fish_pos}"
                else: current_name = f"{fish_id}" # brightfield
                
                assert current_name != previous_name, f"Fish repeated!, check '{current_name}' "
                
                """ comparing """
                if current_name == expect_name:
                    """ True """
                    path = results.pop(0)
                    dname.resave_result(path, recollect_dir)
                    previous_name = current_name
                    if log_mode == "finding": summary[log_mode].append(f"{expect_name}")
                else:
                    """ False """
                    if log_mode == "missing": summary[log_mode].append(f"{expect_name}")

        summary[f"len({log_mode})"] = len(summary[log_mode])        
        
        """ Dump `summary` dict """
        log_path = recollect_dir.joinpath(f"{{Logs}}_collect_{image_type}_results.log")
        with open(log_path, mode="w") as f_writer:
            json.dump(summary, f_writer, indent=4)
        self._logger.info(json.dumps(summary, indent=4))
        
        """ Update `recollect_dir` """
        self._set_recollect_dirs()
    
    
    
    def create_data_xlsx(self):
        """ Create a XLSX file contains `dname` and `brightfield analyze` informations \
            ( used to compute the classes of classification )
        """
        # -----------------------------------------------------------------------------------
        # brightfield
        
        """ Scan `AutoAnalysis`, `ManualAnalysis` results """
        bf_auto_results_list = self.get_sorted_results("brightfield", "AutoAnalysis")[1]
        bf_manual_results_list = self.get_sorted_results("brightfield", "ManualAnalysis")[1]
        self._logger.info((f"brightfield: found "
                           f"{len(bf_auto_results_list)} AutoAnalysis.csv, "
                           f"{len(bf_manual_results_list)} ManualAnalysis.csv, "
                           f"Total: {len(bf_auto_results_list) + len(bf_manual_results_list)} files"))

        """ Merge `AutoAnalysis`, `ManualAnalysis` results """
        bf_auto_results_dict = dname.create_dict_by_id(bf_auto_results_list)
        bf_manual_results_dict = dname.create_dict_by_id(bf_manual_results_list)
        bf_merge_results_dict = dname.merge_dict_by_id(bf_auto_results_dict, bf_manual_results_dict)
        bf_merge_results_list = sorted(list(bf_merge_results_dict.values()), key=dname.get_dname_sortinfo)
        self._logger.info(f"--> After Merging , Total: {len(bf_merge_results_list)} files\n")
        
        # -----------------------------------------------------------------------------------
        # palmskin

        palmskin_processed_dname_dirs = list(self.palmskin_processed_dname_dirs_dict.keys())
        self._logger.info(f"palmskin: found {len(palmskin_processed_dname_dirs)} dname directories\n")
        
        # -----------------------------------------------------------------------------------
        # Main process

        xlsx_path = self.instance_root.joinpath("data.xlsx")
        df_xlsx = pd.DataFrame(columns=["BrightField name with Analysis statement (CSV)",
                                        "Anterior (SP8, .tif)", 
                                        "Posterior (SP8, .tif)",
                                        "Trunk surface area, SA (um2)",
                                        "Standard Length, SL (um)"])
        delete_uncomplete_row = True
        max_probable_num = dname.get_dname_sortinfo(bf_merge_results_list[-1])[0]
        
        for i in range(max_probable_num):
            
            one_base_iter_num = i+1 # Make iteration starting number start from 1
            self._logger.info(f"one_base_iter_num : {one_base_iter_num}")
            
            if  one_base_iter_num == dname.get_dname_sortinfo(bf_merge_results_list[0])[0]:
                
                """ Get informations """
                bf_result_path = bf_merge_results_list.pop(0)
                bf_result_path_split = str(bf_result_path).split(os.sep)
                bf_result_dname = bf_result_path_split[-2]
                bf_result_analysis_type = bf_result_path_split[-1].split(".")[0] # `AutoAnalysis` or `ManualAnalysis`
                self._logger.info(f"bf_result_dname : '{bf_result_dname}'")
                self._logger.info(f"analysis_type : '{bf_result_analysis_type}'")
                
                """ Read CSV """
                analysis_csv = pd.read_csv(bf_result_path, index_col=" ")
                assert len(analysis_csv) == 1, f"More than 1 measurement in csv file, file: '{bf_result_path}'"
                
                """ Get surface area """
                surface_area = analysis_csv.loc[1, "Area"]
                self._logger.info(f"surface_area : {surface_area}")
                
                """ Get standard length """
                standard_length = analysis_csv.loc[1, "Feret"]
                self._logger.info(f"standard_length : {standard_length}")
                
                """ Assign value to Dataframe """
                df_xlsx.loc[one_base_iter_num, "BrightField name with Analysis statement (CSV)"] = f"{bf_result_dname}_{bf_result_analysis_type}"
                df_xlsx.loc[one_base_iter_num, "Trunk surface area, SA (um2)"] = surface_area
                df_xlsx.loc[one_base_iter_num, "Standard Length, SL (um)"] = standard_length

            else: df_xlsx.loc[one_base_iter_num] = np.nan # Can't find corresponding analysis result, make an empty row.
            
            
            if f"{one_base_iter_num}_A" in palmskin_processed_dname_dirs[0]:
                palmskin_A_name = palmskin_processed_dname_dirs.pop(0)
                self._logger.info(f"palmskin_A_name : '{palmskin_A_name}'")
                df_xlsx.loc[one_base_iter_num, "Anterior (SP8, .tif)" ] =  palmskin_A_name
            
            
            if f"{one_base_iter_num}_P" in palmskin_processed_dname_dirs[0]:
                palmskin_P_name = palmskin_processed_dname_dirs.pop(0)
                self._logger.info(f"palmskin_P_name : '{palmskin_P_name}'")
                df_xlsx.loc[one_base_iter_num, "Posterior (SP8, .tif)" ] =  palmskin_P_name

            self._logger.info("\n")

        
        if delete_uncomplete_row: df_xlsx.dropna(inplace=True)
        df_xlsx.to_excel(xlsx_path, engine="openpyxl")
        self._set_data_xlsx_path()
    
    
    
    def __repr__(self):
        
        repr_string = "{\n"
        repr_string += f"self.data_processed_root : '{self.data_processed_root}'\n\n"

        repr_string += f"self.instance_root : '{self.instance_root}'\n"
        repr_string += f"self.instance_desc : '{self.instance_desc}'\n"
        repr_string += f"self.instance_name : '{self.instance_name}'\n\n"

        repr_string += f"self.palmskin_processed_dir : '{self.palmskin_processed_dir}'\n"
        repr_string += f"self.palmskin_processed_reminder : '{self.palmskin_processed_reminder}'\n"
        repr_string += f"self.palmskin_processed_config.param : {json.dumps(self.palmskin_processed_config['param'], indent=4)}\n\n"
                
        repr_string += f"self.brightfield_processed_dir : '{self.brightfield_processed_dir}'\n"
        repr_string += f"self.brightfield_processed_reminder : '{self.brightfield_processed_reminder}'\n"
        repr_string += f"self.brightfield_processed_config.param : {json.dumps(self.brightfield_processed_config['param'], indent=4)}\n\n"
                
        repr_string += f"self.palmskin_recollect_dir : '{self.palmskin_recollect_dir}'\n"
        repr_string += f"self.palmskin_recollected_dirs_dict : {json.dumps(list(self.palmskin_recollected_dirs_dict.keys()), indent=4)}\n\n"
                
        repr_string += f"self.brightfield_recollect_dir : '{self.brightfield_recollect_dir}'\n"
        repr_string += f"self.brightfield_recollected_dirs_dict : {json.dumps(list(self.brightfield_recollected_dirs_dict.keys()), indent=4)}\n\n"
                
        repr_string += f"self.data_xlsx_path : '{self.data_xlsx_path}'\n"
                
        repr_string += f"self.clustered_xlsx_dir : '{self.clustered_xlsx_dir}'\n"
        repr_string += f"self.clustered_xlsx_files_dict : {json.dumps(list(self.clustered_xlsx_files_dict.keys()), indent=4)}\n\n"
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