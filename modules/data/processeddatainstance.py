import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from colorama import Back, Fore, Style

from ..assert_fn import *
from ..assert_fn import (assert_0_or_1_processed_dir,
                         assert_0_or_1_recollect_dir)
from ..shared.baseobject import BaseObject
from ..shared.config import load_config
from ..shared.utils import (create_new_dir, exclude_paths, exclude_tmp_paths,
                            get_target_str_idx_in_list)
from . import dname
# -----------------------------------------------------------------------------/


class ProcessedDataInstance(BaseObject):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Processed Data Instance")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.data_processed_root:Union[None, Path] = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("data_processed")
        
        self.instance_root:Union[None, Path] = None
        self.instance_desc:Union[None, str] = None
        self.instance_name:Union[None, str] = None
        
        self.palmskin_processed_dir:Union[None, Path] = None
        self.palmskin_processed_reminder:Union[None, str] = None
        self.palmskin_processed_dname_dirs_dict:Dict[str, Path] = {}
        self.palmskin_processed_config:dict = {}
        
        self.brightfield_processed_dir:Union[None, Path] = None
        self.brightfield_processed_reminder:Union[None, str] = None
        self.brightfield_processed_dname_dirs_dict:Dict[str, Path] = {}
        self.brightfield_processed_config:dict = {}
        
        self.palmskin_recollect_dir:Union[None, Path] = None
        self.palmskin_recollected_dirs_dict:Dict[str, Path] = {}
        
        self.brightfield_recollect_dir:Union[None, Path] = None
        self.brightfield_recollected_dirs_dict:Dict[str, Path] = {}
        
        self.tabular_file:Union[None, Path] = None
        
        self.clustered_file_dir:Union[None, Path] = None
        self.clustered_files_dict:Dict[str, Path] = {}
        
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def parse_config(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]):
        """
        """
        super()._set_attrs(config)
        self._set_instance_root()
        self._set_processed_dirs()
        self._set_processed_dname_dirs_dicts()
        self._set_processed_configs()
        self._set_recollect_dirs()
        self._set_clustered_file_dir()
        self._set_tabular_file()
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        self.instance_desc = self.config["data_processed"]["instance_desc"]
        # ---------------------------------------------------------------------/


    def _set_instance_root(self):
        """ Set below attributes
            1. `self.instance_root`
            3. `self.instance_name`
        """
        self.instance_root = self._path_navigator.processed_data.get_instance_root(self.config, self._cli_out)
        self.instance_name = str(self.instance_root).split(os.sep)[-1]
        # ---------------------------------------------------------------------/


    def _get_processed_dir(self, image_type:str):
        """ Get one of processed directories,
        
        1. `{[palmskin_reminder]}_PalmSkin_preprocess` or
        2. `{[brightfield_reminder]}_BrightField_analyze`
        
        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        """ Assign `target_text` """
        if image_type == "palmskin":
            target_text = "PalmSkin_preprocess"
        elif image_type == "brightfield":
            target_text = "BrightField_analyze"
        else: raise ValueError(f"Can't recognize arg: '{image_type}'")
        
        """ Scan path """
        found_list = list(self.instance_root.glob(f"{{*}}_{target_text}"))
        assert_0_or_1_processed_dir(found_list, target_text)
        
        """ Assign path """
        if found_list:
            processed_dir = found_list[0]
            """ CLI output """
            self._cli_out.write(f"{image_type.capitalize()} Processed Dir: '{processed_dir}'")
        else:
            processed_dir = None
        
        return processed_dir
        # ---------------------------------------------------------------------/


    def _set_processed_dirs(self):
        """ Set below attributes
            1. `self.palmskin_processed_dir`
            2. `self.palmskin_processed_reminder`
            3. `self.brightfield_processed_dir`
            4. `self.brightfield_processed_reminder`
        """
        """ palmskin """
        path = self._get_processed_dir("palmskin")
        if path is not None:
            self.palmskin_processed_dir = path
            self.palmskin_processed_reminder = re.split("{|}", str(path).split(os.sep)[-1])[1]
        else:
            raise FileNotFoundError("Can't find corresponding 'PalmSkin_preprocess' directory, "
                                    "please run `0.2.preprocess_palmskin.py` to preprocess your palmskin images first.\n")
        
        """ brightfield """
        path = self._get_processed_dir("brightfield")
        if path is not None:
            self.brightfield_processed_dir = path
            self.brightfield_processed_reminder = re.split("{|}", str(path).split(os.sep)[-1])[1]
        else:
            raise FileNotFoundError("Can't find corresponding 'BrightField_analyze' directory, "
                                    "please run `0.3.analyze_brightfield.py` to analyze your brightfield images first.\n")
        # ---------------------------------------------------------------------/


    def _scan_processed_dname_dirs(self, image_type:str):
        """

        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only\n")
        
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
        # ---------------------------------------------------------------------/


    def _update_instance_postfixnum(self):
        """
        """
        old_name = self.instance_name
        new_name = f"{{{self.instance_desc}}}_Academia_Sinica_i{len(self.palmskin_processed_dname_dirs_dict)}"
        
        if new_name != old_name:
            self._cli_out.write("Update Instance Postfix Number ...")
            os.rename(self.instance_root, self.data_processed_root.joinpath(new_name))
            self._set_instance_root()
            self._set_processed_dirs()
            self.palmskin_processed_dname_dirs_dict = self._scan_processed_dname_dirs("palmskin")
        # ---------------------------------------------------------------------/


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
        # ---------------------------------------------------------------------/


    def _get_recollect_dir(self, image_type:str):
        """ Get one of recollect directories,
        
        1. `{[palmskin_reminder]}_PalmSkin_reCollection` or
        2. `{[brightfield_reminder]}_BrightField_reCollection`
        
        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        """ Assign `target_text` """
        if image_type == "palmskin":
            target_text = "PalmSkin_reCollection"
        elif image_type == "brightfield":
            target_text = "BrightField_reCollection"
        else: raise ValueError(f"Can't recognize arg: '{image_type}'")
        
        """ Scan path """
        found_list = list(self.instance_root.glob(f"{{*}}_{target_text}"))
        assert_0_or_1_recollect_dir(found_list, target_text)
        
        """ Assign path """
        if found_list:
            recollect_dir = found_list[0]
            """ CLI output """
            self._cli_out.write(f"{image_type.capitalize()} Recollect Dir: '{recollect_dir}'")
        else:
            recollect_dir = None
        
        return recollect_dir
        # ---------------------------------------------------------------------/


    def _update_recollected_dirs_dict(self, image_type:str):
        """

        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only\n")
        
        # reset dict
        setattr(self, f"{image_type}_recollected_dirs_dict", {})
        recollected_dirs_dict:Dict[str, Path] = getattr(self, f"{image_type}_recollected_dirs_dict")
        
        recollect_dir:Union[None, Path] = getattr(self, f"{image_type}_recollect_dir")
        if recollect_dir is not None:
            """ Scan directories """
            found_list = sorted(list(recollect_dir.glob("*")), key=lambda x: str(x))
            for recollected_dir in found_list:
                recollected_name = str(recollected_dir).split(os.sep)[-1]
                recollected_dirs_dict[recollected_name] = recollected_dir
        # ---------------------------------------------------------------------/


    def _set_recollect_dirs(self):
        """ Set below attributes 
            1. `self.palmskin_recollect_dir`
            2. `self.palmskin_recollected_dirs_dict`
            3. `self.brightfield_recollect_dir`
            4. `self.brightfield_recollected_dirs_dict`
        """
        """ palmskin """
        path = self._get_recollect_dir("palmskin")
        if self.palmskin_recollect_dir != path:
            self.palmskin_recollect_dir = path
            self._update_recollected_dirs_dict("palmskin")
        else:
            self._update_recollected_dirs_dict("palmskin")
        
        """ brightfield """
        path = self._get_recollect_dir("brightfield")
        if self.brightfield_recollect_dir != path:
            self.brightfield_recollect_dir = path
            self._update_recollected_dirs_dict("brightfield")
        else:
            self._update_recollected_dirs_dict("brightfield")
        # ---------------------------------------------------------------------/


    def _set_tabular_file(self):
        """ Set below attributes
            1. `self.tabular_file`
        """
        tabular_file = self.instance_root.joinpath("data.csv")
        
        if tabular_file.exists():
            """ CLI output """
            self._cli_out.write(f"Tabular File : '{tabular_file}'")
        else:
            tabular_file = None
        
        self.tabular_file = tabular_file
        # ---------------------------------------------------------------------/


    def _get_clustered_file_dir(self):
        """
        """
        clustered_file_dir = self.instance_root.joinpath("Clustered_File")
        
        if clustered_file_dir.exists():
            """ CLI output """
            self._cli_out.write(f"Clustered File Dir: '{clustered_file_dir}'")
        else:
            clustered_file_dir = None
        
        return clustered_file_dir
        # ---------------------------------------------------------------------/


    def _update_clustered_files_dict(self):
        """
        """
        self.clustered_files_dict = {} # reset dict
        
        if self.clustered_file_dir is not None:
            """ Scan files """
            found_list = sorted(list(self.clustered_file_dir.glob("**/{*}_dataset.csv")), key=lambda x: str(x))
            found_list = exclude_tmp_paths(found_list)
            for file in found_list:
                file_name = str(file).split(os.sep)[-1]
                cluster_desc = re.split("{|}", file_name)[1]
                if cluster_desc in self.clustered_files_dict:
                    raise ValueError(f"Mutlple '{{{cluster_desc}}}_dataset.csv' are found, "
                                     f"please check file uniqueness under: '{self.clustered_file_dir}'\n")
                else:
                    self.clustered_files_dict[cluster_desc] = file
        # ---------------------------------------------------------------------/


    def _set_clustered_file_dir(self):
        """ Set below attributes
            1. `self.clustered_file_dir`
            2. `self.clustered_files_dict`
        """
        path = self._get_clustered_file_dir()
        if self.clustered_file_dir != path:
            self.clustered_file_dir = path
            self._update_clustered_files_dict()
        else:
            self._update_clustered_files_dict()
        # ---------------------------------------------------------------------/


    def _load_processed_config(self, image_type:str):
        """

        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only\n")
        
        if image_type == "palmskin":
            target_text = "palmskin_preprocess"
        elif image_type == "brightfield":
            target_text = "brightfield_analyze"
        
        processed_dir:Path = getattr(self, f"{image_type}_processed_dir")
        config_file = processed_dir.joinpath(f"{target_text}_config.toml")
        if config_file.exists():
            config = load_config(config_file)
        else:
            config = {}
        
        setattr(self, f"{image_type}_processed_config", config)
        # ---------------------------------------------------------------------/


    def _set_processed_configs(self):
        """ Set below attributes
            1. `self.palmskin_processed_config`
            2. `self.brightfield_processed_config`
        """
        self._load_processed_config("palmskin")
        self._load_processed_config("brightfield")
        # ---------------------------------------------------------------------/


    def get_sorted_results(self, image_type:str, result_name:str) -> Tuple[Union[None, str], List[Path]]:
        """

        Args:
            image_type (str): `palmskin` or `brightfield`
            result_name (str): one of results in each zebrafish dname directory

        Returns:
            Tuple[str, List[Path]]: `(relative_path_in_dname_dir, sorted_results)`
        """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only\n")
        
        # sorted_results
        processed_dir:Path = getattr(self, f"{image_type}_processed_dir")
        found_list = list(processed_dir.glob(f"**/{result_name}"))
        found_list = exclude_paths(found_list, ["!~delete"])
        sorted_results = sorted(found_list, key=dname.get_dname_sortinfo)
        
        # rel_path
        if len(sorted_results) > 0:
            path = sorted_results[0]
            rel_path_with_dname = path.relative_to(processed_dir)
            rel_path = os.sep.join(str(rel_path_with_dname).split(os.sep)[1:])
        else:
            rel_path = None
        
        return rel_path, sorted_results
        # ---------------------------------------------------------------------/


    def collect_results(self, config:Union[str, Path]):
        """ 

        Args:
            config (Union[str, Path]): a toml file.

        Raises:
            ValueError: If (config key) `image_type` != 'palmskin' or 'brightfield'.
            ValueError: If (config key) `log_mode` != 'missing' or 'finding'.
            FileExistsError: If target `recollect_dir` exists.
        """        
        self.parse_config(config)
        
        """ Get variable """
        image_type   = self.config["collection"]["image_type"]
        result_name = self.config["collection"]["result_name"]
        log_mode     = self.config["collection"]["log_mode"]
        
        """ Check variable """
        if image_type not in ["palmskin", "brightfield"]:
            raise ValueError(f"image_type: '{image_type}', accept 'palmskin' or 'brightfield' only\n")
            
        if log_mode not in ["missing", "finding"]:
            raise ValueError(f"log_mode = '{log_mode}', accept 'missing' or 'finding' only\n")
        
        """ Scan results """
        rel_path, sorted_results = self.get_sorted_results(image_type, result_name)
        assert rel_path is not None, "Can't find any result file, `result_name` should be a FULL_NAME (with file extension)"
        
        """ Get `recollect_dir` """
        if image_type == "palmskin": target_text = "PalmSkin"
        elif image_type == "brightfield": target_text = "BrightField"
        reminder = getattr(self, f"{image_type}_processed_reminder")
        recollect_dir = self.instance_root.joinpath(f"{{{reminder}}}_{target_text}_reCollection", os.path.splitext(result_name)[0])
        if recollect_dir.exists():
            raise FileExistsError(f"Directory: '{recollect_dir.resolve()}' already exists, please delete it before collecting results.\n")
        else:
            create_new_dir(recollect_dir)
        
        """ Main process """
        summary = {}
        summary["result_name"] = result_name
        summary["relative_path_in_dname_dir"] = rel_path
        summary["max_probable_num"] = dname.get_dname_sortinfo(sorted_results[-1])[0]
        summary["total files"] = len(sorted_results)
        summary[log_mode] = []
        
        previous_name = ""
        for i in range(summary["max_probable_num"]):
            
            one_base_iter_num = i+1
            
            """ pos option """
            if image_type == "palmskin": pos_list = ["A", "P"]
            else: pos_list = [""] # brightfield
            
            for pos in pos_list:
                
                if len(sorted_results) == 0: 
                    # 如果最後一個 ID 沒有 P , `sorted_results` 會在最後一個 ID 的 A 耗盡
                    # 使用 contunue 強制跳過
                    continue
                
                """ expect_name """
                if image_type == "palmskin": expect_name = f"{one_base_iter_num}_{pos}"
                else: expect_name = f"{one_base_iter_num}" # brightfield
                
                """ current_name """
                fish_id, fish_pos = dname.get_dname_sortinfo(sorted_results[0])
                if image_type == "palmskin": current_name = f"{fish_id}_{fish_pos}"
                else: current_name = f"{fish_id}" # brightfield
                
                assert current_name != previous_name, f"Fish repeated!, check '{current_name}' "
                
                """ comparing """
                if current_name == expect_name:
                    """ True """
                    path = sorted_results.pop(0)
                    dname.resave_result(path, recollect_dir)
                    previous_name = current_name
                    if log_mode == "finding": summary[log_mode].append(f"{expect_name}")
                else:
                    """ False """
                    if log_mode == "missing": summary[log_mode].append(f"{expect_name}")

        summary[f"len({log_mode})"] = len(summary[log_mode])
        
        """ Dump `summary` dict """
        log_file = recollect_dir.joinpath(f"{{Logs}}_collect_{image_type}_results.log")
        with open(log_file, mode="w") as f_writer:
            json.dump(summary, f_writer, indent=4)
        self._cli_out.write(json.dumps(summary, indent=4))
        
        """ Update `recollect_dir` """
        self._set_recollect_dirs()
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def create_tabular_file(self, config:Union[str, Path]):
        """ Create a tabular file contains `dname` and `brightfield analyze` informations \
            ( the gernerated file is used to compute the label of classification )

        Args:
            config (Union[str, Path]): a toml file.
        """
        self.parse_config(config)
        # ---------------------------------------------------------------------
        # brightfield
        
        """ Scan `UNetAnalysis`, `ManualAnalysis` results """
        _, bf_auto_results_list = self.get_sorted_results("brightfield", "UNetAnalysis.csv")
        _, bf_manual_results_list = self.get_sorted_results("brightfield", "ManualAnalysis.csv")
        self._cli_out.write((f"brightfield: found "
                             f"{len(bf_auto_results_list)} UNetAnalysis.csv, "
                             f"{len(bf_manual_results_list)} ManualAnalysis.csv, "
                             f"Total: {len(bf_auto_results_list) + len(bf_manual_results_list)} files"))

        """ Merge `UNetAnalysis`, `ManualAnalysis` results """
        bf_auto_results_dict = dname.create_dict_by_id(bf_auto_results_list)
        bf_manual_results_dict = dname.create_dict_by_id(bf_manual_results_list)
        bf_merge_results_dict = dname.merge_dict_by_id(bf_auto_results_dict, bf_manual_results_dict)
        bf_merge_results_list = sorted(list(bf_merge_results_dict.values()), key=dname.get_dname_sortinfo)
        self._cli_out.write(f"  --> After Merging , Total: {len(bf_merge_results_list)} files")
        
        # ---------------------------------------------------------------------
        # palmskin

        palmskin_processed_dname_dirs = list(self.palmskin_processed_dname_dirs_dict.keys())
        self._cli_out.write(f"palmskin: found {len(palmskin_processed_dname_dirs)} dname directories")
        
        # ---------------------------------------------------------------------
        # Main process
        
        df = pd.DataFrame(columns=["Brightfield",
                                   "Analysis Mode",
                                   "Palmskin Anterior (SP8)", 
                                   "Palmskin Posterior (SP8)",
                                   "Trunk surface area, SA (um2)",
                                   "Standard Length, SL (um)"])
        tabular_file = self.instance_root.joinpath("data.csv")
        delete_uncomplete_row = True
        
        self._cli_out.divide()
        max_probable_num = dname.get_dname_sortinfo(bf_merge_results_list[-1])[0]
        for i in range(max_probable_num):
            
            one_base_iter_num = i+1 # Make iteration starting number start from 1
            self._cli_out.write(f"one_base_iter_num : {one_base_iter_num}")
            
            if  one_base_iter_num == dname.get_dname_sortinfo(bf_merge_results_list[0])[0]:
                
                """ Get informations """
                bf_result_file = bf_merge_results_list.pop(0)
                bf_result_file_split = str(bf_result_file).split(os.sep)
                # dname
                target_idx = get_target_str_idx_in_list(bf_result_file_split, "_BrightField_analyze")
                bf_result_dname = bf_result_file_split[target_idx+1]
                # analysis mode
                bf_result_analysis_mode = os.path.splitext(bf_result_file_split[-1])[0] # `UNetAnalysis` or `ManualAnalysis`
                
                self._cli_out.write(f"bf_result_dname : '{bf_result_dname}'")
                self._cli_out.write(f"analysis_mode : '{bf_result_analysis_mode}'")
                
                """ Read CSV """
                analysis_csv = pd.read_csv(bf_result_file, index_col=" ")
                assert len(analysis_csv) == 1, f"More than 1 measurement in csv file, file: '{bf_result_file}'"
                
                """ Get surface area """
                surface_area = analysis_csv.loc[1, "Area"]
                self._cli_out.write(f"surface_area : {surface_area}")
                
                """ Get standard length """
                standard_length = analysis_csv.loc[1, "Feret"]
                self._cli_out.write(f"standard_length : {standard_length}")
                
                """ Assign value to Dataframe """
                df.loc[one_base_iter_num, "Brightfield"] = bf_result_dname
                df.loc[one_base_iter_num, "Analysis Mode"] = bf_result_analysis_mode
                df.loc[one_base_iter_num, "Trunk surface area, SA (um2)"] = surface_area
                df.loc[one_base_iter_num, "Standard Length, SL (um)"] = standard_length

            else: df.loc[one_base_iter_num] = np.nan # Can't find corresponding analysis result, make an empty row.
            
            if len(palmskin_processed_dname_dirs) > 0:
                
                # position A
                sortinfo = dname.get_dname_sortinfo(palmskin_processed_dname_dirs[0])
                if f"{one_base_iter_num}_A" == f"{sortinfo[0]}_{sortinfo[1]}":
                    palmskin_A_name = palmskin_processed_dname_dirs.pop(0)
                    self._cli_out.write(f"palmskin_A_name : '{palmskin_A_name}'")
                    df.loc[one_base_iter_num, "Palmskin Anterior (SP8)" ] =  palmskin_A_name
                
                # position P
                sortinfo = dname.get_dname_sortinfo(palmskin_processed_dname_dirs[0])
                if f"{one_base_iter_num}_P" == f"{sortinfo[0]}_{sortinfo[1]}":
                    palmskin_P_name = palmskin_processed_dname_dirs.pop(0)
                    self._cli_out.write(f"palmskin_P_name : '{palmskin_P_name}'")
                    df.loc[one_base_iter_num, "Palmskin Posterior (SP8)" ] =  palmskin_P_name
            
            self._cli_out.divide()

        
        if delete_uncomplete_row: df.dropna(inplace=True)
        df.to_csv(tabular_file, encoding='utf_8_sig')
        self._set_tabular_file()
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def check_palmskin_images_condition(self, config:Union[str, Path]):
        """ Check the existence and readability of the palmskin images recorded in the XLSX file.

        Args:
            config (Union[str, Path]): a toml file.

        Raises:
            RuntimeError: If detect a broken/non-existing image.
        """
        self._cli_out._display_on_CLI = False # close CLI output temporarily
        self.parse_config(config)
        self._cli_out._display_on_CLI = True
        
        """ Get variable """
        palmskin_result_name = self.config["data_processed"]["palmskin_result_name"]
        
        """ Get dnames record in CSV  """
        if self.tabular_file is None:
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find tabular file, "
                                    f"please run `0.5.1.create_tabular_file.py` to create it. {Style.RESET_ALL}\n")
        df: pd.DataFrame = pd.read_csv(self.tabular_file, encoding='utf_8_sig')
        palmskin_dnames = sorted(pd.concat([df["Palmskin Anterior (SP8)"], df["Palmskin Posterior (SP8)"]]), key=dname.get_dname_sortinfo)
        
        """ Get specific results exist in 'PalmSkin_preprocess' directory """
        _, sorted_results = self.get_sorted_results("palmskin", palmskin_result_name)
        
        # sorted_results_dict
        sorted_results_dict:Dict[str, Path] = {}
        for path in sorted_results:
            path_split = str(path).split(os.sep)
            target_idx = get_target_str_idx_in_list(path_split, "_PalmSkin_preprocess")
            palmskin_dname = path_split[target_idx+1]
            sorted_results_dict[palmskin_dname] = path
        
        """ Main Process """
        self._cli_out.divide()
        self._reset_pbar()
        with self._pbar:
            
            # add task to `self._pbar`
            task_desc = f"[yellow]Check Image Condition: "
            task = self._pbar.add_task(task_desc, total=len(palmskin_dnames))
            
            read_failed = 0
            for palmskin_dname in palmskin_dnames:
                dyn_desc = f"[yellow]Check Image Condition ( {palmskin_dname} ) : "
                self._pbar.update(task, description=dyn_desc)
                self._pbar.refresh()
                try:
                    palmskin_result_file = sorted_results_dict.pop(palmskin_dname)
                    if cv2.imread(str(palmskin_result_file)) is None:
                        read_failed += 1
                        self._cli_out.write(f"{Fore.RED}{Back.BLACK}Can't read '{palmskin_result_name}' "
                                            f"of '{palmskin_dname}'{Style.RESET_ALL}")
                except KeyError:
                    read_failed += 1
                    self._cli_out.write(f"{Fore.RED}{Back.BLACK}Can't find '{palmskin_result_name}' "
                                        f"of '{palmskin_dname}'{Style.RESET_ALL}")
                self._pbar.update(task, advance=1)
                self._pbar.refresh()
        
        self._cli_out.new_line()
        if read_failed == 0: self._cli_out.write(f"Check Image Condition: {Fore.GREEN}Passed{Style.RESET_ALL}")
        else: raise RuntimeError(f"{Fore.RED} Due to broken/non-existing images, the process has been halted. {Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/