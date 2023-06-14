import os
import sys
import traceback
import re
from colorama import Fore, Back, Style
from pathlib import Path
from typing import List, Dict, Tuple, Union
from collections import OrderedDict
import json
import toml
from logging import Logger

import numpy as np
import pandas as pd
import cv2

abs_module_path = Path("./../../modules/").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from logger import init_logger
from fileop import create_new_dir, resave_result
from data.utils import get_fish_id_pos, create_dict_by_fishid, merge_bf_analysis


class ProcessedDataInstance():
    
    def __init__(self, config_dir:Path, instance_desc:str) -> None:
        
        if isinstance(config_dir, Path): 
            self.config_dir = config_dir
        else: raise TypeError(f"`config_dir` should be a `Path` object, please using `from pathlib import Path`")
        
        # -----------------------------------------------------------------------------------
        # Load `db_path_plan.toml`
        
        with open(config_dir.joinpath("db_path_plan.toml"), mode="r") as f_reader:
            self.dbpp_config = toml.load(f_reader)
        self.db_root = Path(self.dbpp_config["root"])
        
        # -----------------------------------------------------------------------------------
        # variables
        self.data_processed_root:Union[None, Path] = None

        self.instance_desc = instance_desc
        self.instance_root:Union[None, Path] = None
        self.instance_name:Union[None, str]  = None
        
        self.palmskin_preprocess_dir:Union[None, Path]                       = None
        self.palmskin_preprocess_reminder:Union[None, str]                   = None
        self.palmskin_preprocess_config:Union[None, Path]                    = None
        self.palmskin_preprocess_alias_map:Union[None, Dict[str, str]]       = None
        self.palmskin_preprocess_fish_dirs_dict:Union[None, Dict[str, Path]] = None
        
        self.brightfield_analyze_dir:Union[None, Path]                       = None
        self.brightfield_analyze_reminder:Union[None, str]                   = None
        self.brightfield_analyze_config:Union[None, Path]                    = None
        self.brightfield_analyze_alias_map:Union[None, Dict[str, str]]       = None
        self.brightfield_analyze_fish_dirs_dict:Union[None, Dict[str, Path]] = None
        
        self.palmskin_recollect_dir:Union[None, Path]                    = None
        self.palmskin_recollected_dirs_dict:Union[None, Dict[str, Path]] = None
        
        self.brightfield_recollect_dir:Union[None, Path]                    = None
        self.brightfield_recollected_dirs_dict:Union[None, Dict[str, Path]] = None
        
        self.data_xlsx_path:Union[None, Path] = None
        
        self.clustered_xlsx_dir:Union[None, Path] = None
        self.clustered_xlsx_paths_dict:Union[None, Dict[str, Path]] = None
        
        # -----------------------------------------------------------------------------------
        # Check uniqueness of `instance_desc` ( full directory name: `{instance_desc}_Academia_Sinica_i[num]` )
        
        self._find_instance_dir()
        
        # -----------------------------------------------------------------------------------
        # Check direcotry: `{reminder}_PalmSkin_preprocess`
        
        self._check_processed_dir("PalmSkin_preprocess")
        self._get_existing_processed_fish_dirs("PalmSkin_preprocess")
        if self.instance_name.split("_")[-1] == "iTBA": self._update_instance_num_postfix()
        self._init_palmskin_preprocess_alias_map()
        
        # -----------------------------------------------------------------------------------
        # Check direcotry: `{reminder}_BrightField_analyze`, 
        
        self._check_processed_dir("BrightField_analyze")
        self._get_existing_processed_fish_dirs("BrightField_analyze")
        self._init_brightfield_analyze_alias_map()

        # -----------------------------------------------------------------------------------
        # Check direcotries: `{reminder}_PalmSkin_reCollection`, `{reminder}_BrightField_reCollection`
        
        self._check_recollect_dir("PalmSkin")
        self._check_recollect_dir("BrightField")
    
        # -----------------------------------------------------------------------------------
        # Check file: `data.xlsx`
        
        self._check_data_xlsx_path()
        
        # -----------------------------------------------------------------------------------
        # Check direcotry: `Clustered_xlsx`
        
        self._check_clustered_xlsx_dir()
        
        # -----------------------------------------------------------------------------------
        #  TODO:  Check direcotry: `Modified_xlsx`
        
        # self._check_modified_xlsx_dir()  (func name TBA)
        
        # End section -----------------------------------------------------------------------------------
    
    
    
    def _find_instance_dir(self):
        self.data_processed_root = self.db_root.joinpath(self.dbpp_config["data_processed"])
        candidate_dir_list = list(self.data_processed_root.glob(f"*{self.instance_desc}*"))
        assert len(candidate_dir_list) == 1, (f"the given '{self.instance_desc}' is not unique/exists in '({self.data_processed_root}', "
                                              f"find {len(candidate_dir_list)} possible directories, {candidate_dir_list}")
        self.instance_root = candidate_dir_list[0]
        self.instance_name = str(self.instance_root).split(os.sep)[-1]
    
    
    
    def _check_processed_dir(self, processed_name:str):
        """

        Args:
            processed_name (str): `'BrightField_analyze'` or `'PalmSkin_preprocess'`
        """
        assert (processed_name == "BrightField_analyze") or (processed_name == "PalmSkin_preprocess"), \
            f"processed_name = '{processed_name}', accept 'BrightField_analyze' or 'PalmSkin_preprocess' only"
        
        candidate_dir_list = list(self.instance_root.glob(f"*{processed_name}*"))
        assert len(candidate_dir_list) == 1, (f"found {len(candidate_dir_list)} compatible directories, "
                                              f"there must be one and only one `{processed_name}` in '{self.instance_root}'.")
        
        setattr(self, f"{processed_name.lower()}_dir", candidate_dir_list[0])
        setattr(self, f"{processed_name.lower()}_reminder", re.split("{|}", str(getattr(self, f"{processed_name.lower()}_dir")).split(os.sep)[-1])[1])
    
    
    
    def _get_existing_processed_fish_dirs(self, processed_name:str):
        """

        Args:
            processed_name (str): `'BrightField_analyze'` or `'PalmSkin_preprocess'`
        """
        assert (processed_name == "BrightField_analyze") or (processed_name == "PalmSkin_preprocess"), \
            f"processed_name = '{processed_name}', accept 'BrightField_analyze' or 'PalmSkin_preprocess' only"
        processed_name_lower = processed_name.lower()
        
        processed_dir:Path = getattr(self, f"{processed_name_lower}_dir")
        fish_dirs = processed_dir.glob("*")
        
        fish_dirs_dict = {str(fish_dir).split(os.sep)[-1]: fish_dir
                                          for fish_dir in fish_dirs}
        
        for key in list(fish_dirs_dict.keys()):
            if key == "!~delete": fish_dirs_dict.pop(key) # rm "!~delete" directory
            if ".log" in key: fish_dirs_dict.pop(key) # rm "log" files
            if ".toml" in key: fish_dirs_dict.pop(key) # rm "toml" file
        
        fish_dirs_dict = OrderedDict(sorted(list(fish_dirs_dict.items()), key=lambda x: get_fish_id_pos(x[0])))
        
        setattr(self, f"{processed_name_lower}_fish_dirs_dict", fish_dirs_dict)
    
    
    
    def _update_instance_num_postfix(self):
        
        new_name = f"{{{self.instance_desc}}}_Academia_Sinica_i{len(self.palmskin_preprocess_fish_dirs_dict)}"
        os.rename(self.instance_root, self.data_processed_root.joinpath(new_name))
        
        self.instance_root = Path(str(self.instance_root).replace(self.instance_name, new_name))
        self.palmskin_preprocess_dir = Path(str(self.palmskin_preprocess_dir).replace(self.instance_name, new_name))
        self.instance_name = new_name
    
    
    
    def _check_recollect_dir(self, source_name:str):
        """

        Args:
            source_name (str): `'BrightField'` or `'PalmSkin'`
        """
        assert (source_name == "BrightField") or (source_name == "PalmSkin"), \
            f"source_name = '{source_name}', accept 'BrightField' or 'PalmSkin' only"
        source_name_lower = source_name.lower()
        
        candidate_dir_list = list(self.instance_root.glob(f"*{source_name}_reCollection*"))
        assert len(candidate_dir_list) <= 1, (f"found {len(candidate_dir_list)} compatible directories, only one `{source_name}_reCollection` is accepted.")
        if len(candidate_dir_list) == 1: 
            setattr(self, f"{source_name_lower}_recollect_dir", candidate_dir_list[0])
            self._update_recollected_dirs_dict(source_name)
    
    
    
    def _update_recollected_dirs_dict(self, source_name:str):
        """

        Args:
            source_name (str): `'BrightField'` or `'PalmSkin'`
        """
        assert (source_name == "BrightField") or (source_name == "PalmSkin"), \
            f"source_name = '{source_name}', accept 'BrightField' or 'PalmSkin' only"
        source_name_lower = source_name.lower()
        
        recollect_dir:Path = getattr(self, f"{source_name_lower}_recollect_dir")
        if getattr(self, f"{source_name_lower}_recollected_dirs_dict") is None:
            setattr(self, f"{source_name_lower}_recollected_dirs_dict", {})
        recollected_dict:Union[None, Dict[str, Path]] = getattr(self, f"{source_name_lower}_recollected_dirs_dict")
        
        scaned_list = list(recollect_dir.glob("*"))
        for dir in scaned_list:
            dir_name = str(dir).split(os.sep)[-1]
            try:
                recollected_dict[dir_name]
            except:
                recollected_dict[dir_name] = dir
    
    
    
    def _check_data_xlsx_path(self):
        path = self.instance_root.joinpath("data.xlsx")
        if path.exists(): self.data_xlsx_path = path
    
    
    
    def _check_clustered_xlsx_dir(self):
        candidate_dir_list = list(self.instance_root.glob(f"Clustered_xlsx"))
        assert len(candidate_dir_list) <= 1, (f"found {len(candidate_dir_list)} compatible directories, only one `Clustered_xlsx` is accepted.")
        if len(candidate_dir_list) == 1: 
            self.clustered_xlsx_dir = candidate_dir_list[0]
            self._update_clustered_xlsx_paths_dict()
    
    
    
    def _update_clustered_xlsx_paths_dict(self):
        
        if self.clustered_xlsx_paths_dict is None:
            self.clustered_xlsx_paths_dict = {}
        
        scaned_list = list(self.clustered_xlsx_dir.glob("*.xlsx"))
        for xlsx_path in scaned_list:
            xlsx_name = str(xlsx_path).split(os.sep)[-1]
            cluster_desc = re.split("{|}", xlsx_name)[1]
            try:
                self.clustered_xlsx_paths_dict[cluster_desc]
            except:
                self.clustered_xlsx_paths_dict[cluster_desc] = xlsx_path
    
    
    
    def _init_palmskin_preprocess_alias_map(self):
        
        with open(self.palmskin_preprocess_dir.joinpath("palmskin_preprocess_config.toml"), mode="r") as f_reader:
            self.palmskin_preprocess_config = toml.load(f_reader)
        
        preprocess_kwargs = self.palmskin_preprocess_config["param"]
        Kuwahara = f"Kuwahara{preprocess_kwargs['Kuwahara_sampleing']}"
        bf_zproj_type = f"BF_Zproj_{preprocess_kwargs['bf_zproj_type']}"
        bf_threshold = f"0_{preprocess_kwargs['bf_treshold_value']}"
        
        self.palmskin_preprocess_alias_map = {
            "RGB_direct_max_zproj":          "*_RGB_direct_max_zproj.tif", # CHECK_PT 
            # -----------------------------------------------------------------------------------
            "ch_B":                          "MetaImage/*_B_processed.tif",
            "ch_B_Kuwahara":                 f"MetaImage/*_B_processed_{Kuwahara}.tif",
            "ch_B_fusion":                   "*_B_processed_fusion.tif", # CHECK_PT 
            "ch_B_HE":                       "MetaImage/*_B_processed_HE.tif",
            "ch_B_Kuwahara_HE":              f"MetaImage/*_B_processed_{Kuwahara}_HE.tif",
            "ch_B_HE_fusion":                "*_B_processed_HE_fusion.tif", # CHECK_PT 
            # -----------------------------------------------------------------------------------
            "ch_G":                          "MetaImage/*_G_processed.tif",
            "ch_G_Kuwahara":                 f"MetaImage/*_G_processed_{Kuwahara}.tif",
            "ch_G_fusion":                   "*_G_processed_fusion.tif", # CHECK_PT 
            "ch_G_HE":                       "MetaImage/*_G_processed_HE.tif",
            "ch_G_Kuwahara_HE":              f"MetaImage/*_G_processed_{Kuwahara}_HE.tif",
            "ch_G_HE_fusion":                "*_G_processed_HE_fusion.tif", # CHECK_PT 
            # -----------------------------------------------------------------------------------
            "ch_R":                          "MetaImage/*_R_processed.tif",
            "ch_R_Kuwahara":                 f"MetaImage/*_R_processed_{Kuwahara}.tif",
            "ch_R_fusion":                   "*_R_processed_fusion.tif", # CHECK_PT 
            "ch_R_HE":                       "MetaImage/*_R_processed_HE.tif",
            "ch_R_Kuwahara_HE":              f"MetaImage/*_R_processed_{Kuwahara}_HE.tif",
            "ch_R_HE_fusion":                "*_R_processed_HE_fusion.tif", # CHECK_PT 
            # -----------------------------------------------------------------------------------
            "RGB":                           "MetaImage/*_RGB_processed.tif",
            "RGB_Kuwahara":                  f"MetaImage/*_RGB_processed_{Kuwahara}.tif",
            "RGB_fusion":                    "*_RGB_processed_fusion.tif", # CHECK_PT  => Average(RGB_processed, RGB_processed_Kuwahara)
            "RGB_fusion2Gray":               "*_RGB_processed_fusion2Gray.tif", # CHECK_PT 
            "RGB_HE" :                       "MetaImage/*_RGB_processed_HE.tif",
            "RGB_Kuwahara_HE" :              f"MetaImage/*_RGB_processed_{Kuwahara}_HE.tif",
            "RGB_HE_fusion" :                "*_RGB_processed_HE_fusion.tif", # CHECK_PT  => Average(RGB_processed_HE, RGB_processed_Kuwahara_HE)
            "RGB_HE_fusion2Gray":            "*_RGB_processed_HE_fusion2Gray.tif", # CHECK_PT 
            # -----------------------------------------------------------------------------------
            "BF_Zproj":                      f"MetaImage/*_{bf_zproj_type}.tif",
            "BF_Zproj_HE":                   f"MetaImage/*_{bf_zproj_type}_HE.tif",
            "Threshold":                     f"MetaImage/*_Threshold_{bf_threshold}.tif",
            "outer_rect":                    "MetaImage/*_outer_rect.tif",
            "inner_rect":                    "MetaImage/*_inner_rect.tif",
            "RoiSet" :                       "MetaImage/RoiSet_AutoRect.roi",
            # -----------------------------------------------------------------------------------
            "RGB_fusion--AutoRect":          "*_RGB_processed_fusion--AutoRect.tif", # CHECK_PT 
            "RGB_HE_fusion--AutoRect":       "*_RGB_processed_HE_fusion--AutoRect.tif", # CHECK_PT 
            # -----------------------------------------------------------------------------------
            "autocropped_RGB_fusion" :       "*_autocropped_RGB_processed_fusion.tif", # CHECK_PT 
            "autocropped_RGB_HE_fusion" :    "*_autocropped_RGB_processed_HE_fusion.tif", # CHECK_PT 
        }
    
    
    
    def _init_brightfield_analyze_alias_map(self):
        
        with open(self.brightfield_analyze_dir.joinpath("brightfield_analyze_config.toml"), mode="r") as f_reader:
            self.brightfield_analyze_config = toml.load(f_reader)
        
        analyze_kwargs = self.brightfield_analyze_config["param"]
        autothreshold_algo = analyze_kwargs['auto_threshold']
        
        self.brightfield_analyze_alias_map = {
            "original_16bit" :          "MetaImage/*_original_16bit.tif",
            "cropped_BF" :              "*_cropped_BF.tif", # CHECK_PT 
            "AutoThreshold" :           f"MetaImage/*_AutoThreshold_{autothreshold_algo}.tif",
            "measured_mask" :           "MetaImage/*_measured_mask.tif",
            "cropped_BF--MIX" :         "*_cropped_BF--MIX.tif", # CHECK_PT 
            "RoiSet" :                  "MetaImage/RoiSet.zip",
            "AutoAnalysis" :            "AutoAnalysis.csv",
            "ManualAnalysis" :          "ManualAnalysis.csv",
            "Manual_measured_mask" :    "Manual_measured_mask.tif", # CHECK_PT 
            "Manual_cropped_BF--MIX" :  "Manual_cropped_BF--MIX.tif", # CHECK_PT 
        }
    
    
    
    def get_existing_processed_results(self, processed_name:str, result_alias:str) -> Tuple[str, List[Path]]:
        """

        Args:
            processed_name (str): `'BrightField_analyze'` or `'PalmSkin_preprocess'`
            result_alias (str): please refer to `'Documents/{NamingRule}_ResultAlias.md'` in this repository

        Returns:
            Tuple[str, List[Path]]: `(actual_name, results)`
        """
        assert (processed_name == "BrightField_analyze") or (processed_name == "PalmSkin_preprocess"), \
            f"processed_name = '{processed_name}', accept 'BrightField_analyze' or 'PalmSkin_preprocess' only"
        processed_name_lower = processed_name.lower()
        
        processed_dir:Path = getattr(self, f"{processed_name_lower}_dir")
        alias_map = getattr(self, f"{processed_name_lower}_alias_map")
        assert alias_map[result_alias]
        
        # regex filter
        results = sorted(processed_dir.glob(f"*/{alias_map[result_alias]}"), key=get_fish_id_pos)
        pattern = alias_map[result_alias].split("/")[-1]
        pattern = pattern.replace("*", r"[0-9]*")
        num = 0
        actual_name = None
        for _ in range(len(results)):
            result_name = str(results[num]).split(os.sep)[-1]
            if not re.fullmatch(pattern, result_name):
                results.pop(num)
            else:
                num += 1
                if actual_name is None: actual_name = result_name
            
        return actual_name, results
    
    
    
    def collect_results(self, processed_name:str, result_alias:str, log_mode:str="missing"):
        """

        Args:
            processed_name (str): `'BrightField_analyze'` or `'PalmSkin_preprocess'`
            result_alias (str): please refer to `'Documents/{NamingRule}_ResultAlias.md'` in this repository
            log_mode (str, optional): `'missing'` or `'finding'`. Defaults to "missing".
        """
        assert (processed_name == "BrightField_analyze") or (processed_name == "PalmSkin_preprocess"), \
            f"processed_name = '{processed_name}', accept 'BrightField_analyze' or 'PalmSkin_preprocess' only"
            
        assert (log_mode == "missing") or (log_mode == "finding"), \
            f"log_mode = '{log_mode}', accept 'missing' or 'finding' only"
        
        processed_name_lower = processed_name.lower()
        source_name = processed_name.split("_")[0]
        source_name_lower = source_name.lower()
        
        # get attributes
        processed_reminder = getattr(self, f"{processed_name_lower}_reminder")
        alias_map = getattr(self, f"{processed_name_lower}_alias_map")
        recollect_dir:Path = getattr(self, f"{source_name_lower}_recollect_dir")
        assert alias_map[result_alias]
        
        # output
        output_dir = self.instance_root.joinpath(f"{{{processed_reminder}}}_{source_name}_reCollection", result_alias)
        assert not output_dir.exists(), f"Directory: '{output_dir}' already exists, please delete it before collecting results."
        create_new_dir(output_dir)
        
        actual_name, results = self.get_existing_processed_results(processed_name, result_alias)
        
        summary = {}
        summary["result_alias"] = result_alias
        summary["actual_name"] = actual_name
        summary["max_probable_num"] = get_fish_id_pos(results[-1])[0]
        summary["total files"] = len(results)
        summary[log_mode] = []
        
        previous_fish = ""
        for i in range(summary["max_probable_num"]):
            
            one_base_iter_num = i+1
            
            if source_name == "PalmSkin": pos_list = ["A", "P"]
            else: pos_list = [""]
            
            
            for pos in pos_list:
                
                # expect_name
                if source_name == "PalmSkin": expect_name = f"{one_base_iter_num}_{pos}"
                else: expect_name = f"{one_base_iter_num}" # BrightField
                
                try: # current_name
                    
                    fish_ID, fish_pos = get_fish_id_pos(results[0])
                    if source_name == "PalmSkin": current_name = f"{fish_ID}_{fish_pos}"
                    else: current_name = f"{fish_ID}" # BrightField
                    assert current_name != previous_fish, f"fish_dir repeated!, check '{previous_fish}' "
                
                except: pass
                
                # comparing
                if current_name == expect_name:
                    
                    path = results.pop(0)
                    resave_result(path, output_dir, alias_map[result_alias])
                    previous_fish = current_name
                    if log_mode == "finding": summary[log_mode].append(f"{expect_name}")
                    
                else: 
                    if log_mode == "missing": summary[log_mode].append(f"{expect_name}")

        
        summary[f"len({log_mode})"] = len(summary[log_mode])
        print(json.dumps(summary, indent=4))
        
        # write log
        log_path = output_dir.joinpath(f"{{Logs}}_collect_{processed_name_lower}_results.log")
        with open(log_path, mode="w") as f_writer:
            json.dump(summary, f_writer, indent=4)
        
        # update `recollect_dir`
        if recollect_dir is None: self._check_recollect_dir(source_name)
        else: self._update_recollected_dirs_dict(source_name)
    
    
    
    def create_data_xlsx(self, logger:Logger):
        """To generate data information in XLSX ( XLSX file will used to compute the classes in classification task ):

            All fish will process with the following step : 
            
                1. Run ImageJ Macro : Use bright field (BF) images to compute the surface area (SA) and surface length (SL), and store their results in CSV format.
                2. Collect all generated CSV files using pandas.DataFrame().
                3. Use `fish_id` to find and add their `palmskin_RGB` images into the DataFrame.
                4. Save results in XLSX format.

        Args:
            logger (Logger): external logger created using package `logging`
        """
        
        # -----------------------------------------------------------------------------------
        # BrightField
        
        # Scan `AutoAnalysis` results, and sort ( Due to OS scanning strategy 10 may listed before 8 )
        bf_recollect_auto_list = self.get_existing_processed_results("BrightField_analyze", "AutoAnalysis")[1]
        bf_recollect_auto_list = sorted(bf_recollect_auto_list, key=get_fish_id_pos)

        # Scan `ManualAnalysis` results, and sort ( Due to OS scanning strategy 10 may listed before 8 )
        bf_recollect_manual_list = self.get_existing_processed_results("BrightField_analyze", "ManualAnalysis")[1]
        bf_recollect_manual_list = sorted(bf_recollect_manual_list, key=get_fish_id_pos)

        # show info
        logger.info((f"BrightField: Found {len(bf_recollect_auto_list)} AutoAnalysis.csv, "
                f"{len(bf_recollect_manual_list)} ManualAnalysis.csv, "
                f"Total: {len(bf_recollect_auto_list) + len(bf_recollect_manual_list)} files"))

        # Merge `AutoAnalysis` and `ManualAnalysis` list
        bf_recollect_auto_dict = create_dict_by_fishid(bf_recollect_auto_list)
        bf_recollect_manual_dict = create_dict_by_fishid(bf_recollect_manual_list)
        bf_recollect_merge_dict = merge_bf_analysis(bf_recollect_auto_dict, bf_recollect_manual_dict)
        bf_recollect_merge_list = sorted(list(bf_recollect_merge_dict.values()), key=get_fish_id_pos)
        logger.info(f"--> After Merging , Total: {len(bf_recollect_merge_list)} files")
        
        # -----------------------------------------------------------------------------------
        # PalmSkin

        palmskin_preprocess_fish_dirs = list(self.palmskin_preprocess_fish_dirs_dict.keys())
        logger.info(f"PalmSkin: Found {len(palmskin_preprocess_fish_dirs)} tif files")
        
        # -----------------------------------------------------------------------------------
        # Processing

        delete_uncomplete_row = True
        output = os.path.join(self.instance_root, r"data.xlsx")

        # Creating "data.xlsx"
        data = pd.DataFrame(columns=["BrightField name with Analysis statement (CSV)",
                                    "Anterior (SP8, .tif)", 
                                    "Posterior (SP8, .tif)",
                                    "Trunk surface area, SA (um2)",
                                    "Standard Length, SL (um)"])


        print("\n\nprocessing...\n")

        # Variable
        max_probable_num = get_fish_id_pos(bf_recollect_merge_list[-1])[0]
        logger.info(f'max_probable_num {type(max_probable_num)}: {max_probable_num}\n')


        # Starting...
        for i in range(max_probable_num):
            
            # *** Print CMD section divider ***
            print("="*100, "\n")
            
            one_base_iter_num = i+1 # Make iteration starting number start from 1
            logger.info(f'one_base_iter_num {type(one_base_iter_num)}: {one_base_iter_num}\n')
            
            
            if  one_base_iter_num == get_fish_id_pos(bf_recollect_merge_list[0])[0] :
                
                # Get info strings
                bf_result_path = bf_recollect_merge_list.pop(0)
                bf_result_path_split = str(bf_result_path).split(os.sep)
                bf_result_name = bf_result_path_split[-2] # `AutoAnalysis` or `ManualAnalysis`
                bf_result_analysis_type = bf_result_path_split[-1].split(".")[0] # Get name_noExtension
                logger.info(f'bf_result_name {type(bf_result_name)}: {bf_result_name}')
                logger.info(f'analysis_type {type(bf_result_analysis_type)}: {bf_result_analysis_type}')
                # Read CSV
                analysis_csv = pd.read_csv(bf_result_path, index_col=" ")
                assert len(analysis_csv) == 1, f"More than 1 measure data in csv file, file:{bf_result_path}"
                # Get surface area from analysis file
                surface_area = analysis_csv.loc[1, "Area"]
                logger.info(f'surface_area {type(surface_area)}: {surface_area}')
                # Get standard length from analysis file
                standard_length = analysis_csv.loc[1, "Feret"]
                logger.info(f'standard_length {type(standard_length)}: {standard_length}')
                
                data.loc[one_base_iter_num, "BrightField name with Analysis statement (CSV)"] = f"{bf_result_name}_{bf_result_analysis_type}"
                data.loc[one_base_iter_num, "Trunk surface area, SA (um2)"] = surface_area
                data.loc[one_base_iter_num, "Standard Length, SL (um)"] = standard_length

            else: data.loc[one_base_iter_num] = np.nan # Can't find corresponding analysis result, make an empty row.
            
            
            if f"{one_base_iter_num}_A" in palmskin_preprocess_fish_dirs[0]:
                palmskin_RGB_A_name = palmskin_preprocess_fish_dirs.pop(0)
                logger.info(f'palmskin_RGB_A_name {type(palmskin_RGB_A_name)}: {palmskin_RGB_A_name}')
                data.loc[one_base_iter_num, "Anterior (SP8, .tif)" ] =  palmskin_RGB_A_name
            
            
            if f"{one_base_iter_num}_P" in palmskin_preprocess_fish_dirs[0]:
                palmskin_RGB_P_name = palmskin_preprocess_fish_dirs.pop(0)
                logger.info(f'palmskin_RGB_P_name {type(palmskin_RGB_P_name)}: {palmskin_RGB_P_name}')
                data.loc[one_base_iter_num, "Posterior (SP8, .tif)" ] =  palmskin_RGB_P_name
            
            
            print("\n\n\n")


        if delete_uncomplete_row: data.dropna(inplace=True)
        data.to_excel(output, engine="openpyxl")

        self._check_data_xlsx_path()
    
    
    
    def __repr__(self):
        
        repr_string = f'self.data_processed_root {type(self.data_processed_root)}: {self.data_processed_root}\n\n'
                
        repr_string += f'self.instance_desc : "{self.instance_desc}"\n'
        repr_string += f'self.instance_root {type(self.instance_root)}: "{self.instance_root}"\n'
        repr_string += f'self.instance_name : "{self.instance_name}"\n\n'
                
        repr_string += f'self.palmskin_preprocess_dir {type(self.palmskin_preprocess_dir)}: "{self.palmskin_preprocess_dir}"\n'
        repr_string += f'self.palmskin_preprocess_reminder : {self.palmskin_preprocess_reminder}\n'
        repr_string += f'self.palmskin_preprocess_config.param : {json.dumps(self.palmskin_preprocess_config["param"], indent=4)}\n\n'
                
        repr_string += f'self.brightfield_analyze_dir {type(self.brightfield_analyze_dir)}: "{self.brightfield_analyze_dir}"\n'
        repr_string += f'self.brightfield_analyze_reminder : {self.brightfield_analyze_reminder}\n'
        repr_string += f'self.brightfield_analyze_config.param : {json.dumps(self.brightfield_analyze_config["param"], indent=4)}\n\n'
                
        repr_string += f'self.palmskin_recollect_dir {type(self.palmskin_recollect_dir)}: "{self.palmskin_recollect_dir}"\n'
        repr_string += f'self.palmskin_recollected_dirs_dict : {json.dumps(list(self.palmskin_recollected_dirs_dict.keys()), indent=4)}\n\n'
                
        repr_string += f'self.brightfield_recollect_dir {type(self.brightfield_recollect_dir)}: "{self.brightfield_recollect_dir}"\n'
        repr_string += f'self.brightfield_recollected_dirs_dict : {json.dumps(list(self.brightfield_recollected_dirs_dict.keys()), indent=4)}\n\n'
                
        repr_string += f'self.data_xlsx_path : "{self.data_xlsx_path}"\n\n'
                
        repr_string += f'self.clustered_xlsx_dir {type(self.clustered_xlsx_dir)}: "{self.clustered_xlsx_dir}"\n'
        repr_string += f'self.clustered_xlsx_paths_dict : {json.dumps(list(self.clustered_xlsx_paths_dict.keys()), indent=4)}\n\n'
        
        return repr_string
    
    
    
    def check_palmskin_images_condition(self, palmskin_result_alias:str, xlsx_name:str=None) -> bool:
        """Check the existence and readability of the palm skin images recorded in the XLSX file.

        Args:
            palmskin_result_alias (str): please refer to `'Documents/{NamingRule}_ResultAlias.md'` in this repository.
            xlsx_name (str, optional): If `None`, use `self.data_xlsx_path`
        """
        if xlsx_name is None:
            assert self.data_xlsx_path is not None, "Can't find `data.xlsx` please use `self.create_data_xlsx()` to create it."
            xlsx_path = self.data_xlsx_path
        
        #  TODO:  xlsx_name is not None, use given xlsx under `Modified_xlsx/`
        
        df_xlsx :pd.DataFrame = pd.read_excel(xlsx_path, engine = 'openpyxl')
        
        palmskin_dnames = list(pd.concat([df_xlsx["Anterior (SP8, .tif)"], df_xlsx["Posterior (SP8, .tif)"]]))
        actual_name, processed_palmskin_results = self.get_existing_processed_results("PalmSkin_preprocess", palmskin_result_alias)
        processed_palmskin_results = {str(result_path).split(os.sep)[-2]: result_path for result_path in processed_palmskin_results}
        
        read_failed = 0
        for dname in palmskin_dnames:
            try:
                path = processed_palmskin_results.pop(dname)
                if cv2.imread(str(path)) is None: 
                    read_failed += 1
                    print(f"{Fore.RED}{Back.BLACK}Can't read '{actual_name}' of '{dname}'{Style.RESET_ALL}")
            except:
                print(f"{Fore.RED}{Back.BLACK}Can't find '{actual_name}' of '{dname}'{Style.RESET_ALL}")
                read_failed += 1
        
        if read_failed == 0: return True
        else: return False