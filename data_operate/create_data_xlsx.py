"""

To generate data information in XLSX ( XLSX file will used to compute the classes in classification task ):

    All fish will process with the following step : 
    
        1. Run ImageJ Macro : Use bright field (BF) images to compute the surface area (SA) and surface length (SL), and store their results in CSV format.
        2. Collect all generated CSV files using pandas.DataFrame().
        3. Use "fish_id" to find and add their "palmskin_RGB" images into the DataFrame.
        4. Save results in XLSX format.

"""
# -----------------------------------------------------------------------------------

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from glob import glob
import toml

import numpy as np
import pandas as pd

rel_module_path = "./../modules/"
sys.path.append( str(Path(rel_module_path).resolve()) ) # add path to scan customized module

from logger import init_logger
from data.utils import get_fish_ID_pos, create_dict_by_fishID, merge_BF_analysis

log = init_logger(r"Create Data Xlsx")


# -----------------------------------------------------------------------------------
# Load `db_path_plan.toml`
with open("./../Config/db_path_plan.toml", mode="r") as f_reader:
    dbpp_config = toml.load(f_reader)
db_root = Path(dbpp_config["root"])
        
# -----------------------------------------------------------------------------------
# Load `(CreateXlsx)_data.toml`
with open("./../Config/(CreateXlsx)_data.toml", mode="r") as f_reader:
    config = toml.load(f_reader)
preprocessed_desc = config["data_preprocessed"]["desc"]
palmskin_result_alias = config["result_alias"]

# -----------------------------------------------------------------------------------
# Generate `path_vars`

# Check `{desc}_Academia_Sinica_i[num]`
data_root = db_root.joinpath(dbpp_config["data_preprocessed"])
target_dir_list = list(data_root.glob(f"*{preprocessed_desc}*"))
assert len(target_dir_list) == 1, (f"[data_preprocessed.desc] in `(CreateXlsx)_data.toml` is not unique/exists, "
                                   f"find {len(target_dir_list)} possible directories, {target_dir_list}")
preprocessed_root = target_dir_list[0]


# -----------------------------------------------------------------------------------
# BrightField : `{reminder}_BrightField_reCollection`

# Check `{reminder}_BrightField_reCollection`
target_dir_list = list(preprocessed_root.glob(f"*BrightField_reCollection*"))
assert len(target_dir_list) == 1, (f"found {len(target_dir_list)} directories, only one `BrightField_reCollection` is accepted.")
bf_recollect_root = target_dir_list[0]
bf_analyzed_reminder = re.split("{|}", str(bf_recollect_root).split(os.sep)[-1])[1]

# Scan `AutoAnalysis` files, and sort ( Due to OS scanning strategy 10 may listed before 8 )
bf_recollect_auto_list = sorted(bf_recollect_root.glob("AutoAnalysis/*.csv"), key=get_fish_ID_pos)
assert len(bf_recollect_auto_list) > 0, f"Can't find directory: `{bf_recollect_root}/AutoAnalysis`, or it is empty."

# Scan `ManualAnalysis` files, and sort ( Due to OS scanning strategy 10 may listed before 8 )
bf_recollect_manual_list = sorted(bf_recollect_root.glob("ManualAnalysis/*.csv"), key=get_fish_ID_pos)

# show info
log.info((f"BrightField_reCollection: Found {len(bf_recollect_auto_list)} AutoAnalysis.csv, "
          f"{len(bf_recollect_manual_list)} ManualAnalysis.csv, "
          f"Total: {len(bf_recollect_auto_list) + len(bf_recollect_manual_list)} files"))

# Merge `AutoAnalysis` and `ManualAnalysis` list
bf_recollect_auto_dict = create_dict_by_fishID(bf_recollect_auto_list)
bf_recollect_manual_dict = create_dict_by_fishID(bf_recollect_manual_list)
bf_recollect_merge_dict = merge_BF_analysis(bf_recollect_auto_dict, bf_recollect_manual_dict)
bf_recollect_merge_list = sorted(list(bf_recollect_merge_dict.values()), key=get_fish_ID_pos)
log.info(f"--> After Merging , Total: {len(bf_recollect_merge_list)} files")


# -----------------------------------------------------------------------------------
# PalmSkin : `{reminder}_PalmSkin_reCollection`

# Check `{reminder}_PalmSkin_reCollection`
target_dir_list = list(preprocessed_root.glob(f"*PalmSkin_reCollection*"))
assert len(target_dir_list) == 1, (f"found {len(target_dir_list)} directories, only one `PalmSkin_reCollection` is accepted.")
palmskin_recollect_root = target_dir_list[0]
palmskin_preprocessed_reminder = re.split("{|}", str(palmskin_recollect_root).split(os.sep)[-1])[1]

# Scan files, and sort ( Due to OS scanning strategy 10 may listed before 8 )
palmskin_recollect_list = sorted(palmskin_recollect_root.glob(f"{palmskin_result_alias}/*.tif"), key=get_fish_ID_pos)
assert len(palmskin_recollect_list) > 0, f"Can't find directory: `{palmskin_recollect_root}/{palmskin_result_alias}`, or it is empty."
log.info(f"PalmSkin_reCollection: Found {len(palmskin_recollect_list)} .tif files")


# -----------------------------------------------------------------------------------
# Processing

delete_uncomplete_row = True
output = os.path.join(preprocessed_root, r"data.xlsx")
# Creating "data.xlsx"
data = pd.DataFrame(columns=["BrightField name with Analysis statement (CSV)",
                             "Anterior (SP8, .tif)", 
                             "Posterior (SP8, .tif)",
                             "Trunk surface area, SA (um2)",
                             "Standard Length, SL (um)"])


print("\n\nprocessing...\n")

# Variable
max_probable_num = get_fish_ID_pos(bf_recollect_merge_list[-1])[0]
log.info(f'max_probable_num {type(max_probable_num)}: {max_probable_num}\n')
bf_result_iter_i = 0
palmskin_RGB_iter_i = 0


# Starting...
for i in range(max_probable_num):
    
    # *** Print CMD section divider ***
    print("="*100, "\n")
    
    one_base_iter_num = i+1 # Make iteration starting number start from 1
    log.info(f'one_base_iter_num {type(one_base_iter_num)}: {one_base_iter_num}\n')
    
    
    if  one_base_iter_num == get_fish_ID_pos(bf_recollect_merge_list[0])[0] :
        
        # Get info strings
        bf_result_path = bf_recollect_merge_list.pop(0)
        bf_result_path_split = str(bf_result_path).split(os.sep)
        bf_result_name = bf_result_path_split[-1].split(".")[0] # Get name_noExtension
        bf_result_analysis_type = bf_result_path_split[-2] # `AutoAnalysis` or `ManualAnalysis`
        log.info(f'bf_result_name {type(bf_result_name)}: {bf_result_name}')
        log.info(f'analysis_type {type(bf_result_analysis_type)}: {bf_result_analysis_type}')
        # Read CSV
        analysis_csv = pd.read_csv(bf_result_path, index_col=" ")
        assert len(analysis_csv) == 1, f"More than 1 measure data in csv file, file:{bf_result_path}"
        # Get surface area from analysis file
        surface_area = analysis_csv.loc[1, "Area"]
        log.info(f'surface_area {type(surface_area)}: {surface_area}')
        # Get standard length from analysis file
        standard_length = analysis_csv.loc[1, "Feret"]
        log.info(f'standard_length {type(standard_length)}: {standard_length}')
        
        data.loc[one_base_iter_num, "BrightField name with Analysis statement (CSV)"] = f"{bf_result_name}_{bf_result_analysis_type}"
        data.loc[one_base_iter_num, "Trunk surface area, SA (um2)"] = surface_area
        data.loc[one_base_iter_num, "Standard Length, SL (um)"] = standard_length

    else: data.loc[one_base_iter_num] = np.nan # Can't find corresponding analysis result, make an empty row.
    
    
    if f"{one_base_iter_num}_A" in str(palmskin_recollect_list[0]):
        palmskin_RGB_A_name = str(palmskin_recollect_list.pop(0)).split(os.sep)[-1].split(".")[0] # Get name_noExtension
        log.info(f'palmskin_RGB_A_name {type(palmskin_RGB_A_name)}: {palmskin_RGB_A_name}')
        data.loc[one_base_iter_num, "Anterior (SP8, .tif)" ] =  palmskin_RGB_A_name
    
    
    if f"{one_base_iter_num}_P" in str(palmskin_recollect_list[0]):
        palmskin_RGB_P_name = str(palmskin_recollect_list.pop(0)).split(os.sep)[-1].split(".")[0] # Get name_noExtension
        log.info(f'palmskin_RGB_P_name {type(palmskin_RGB_P_name)}: {palmskin_RGB_P_name}')
        data.loc[one_base_iter_num, "Posterior (SP8, .tif)" ] =  palmskin_RGB_P_name
    
    
    print("\n\n\n")


if delete_uncomplete_row: data.dropna(inplace=True)
data.to_excel(output, engine="openpyxl")

print("="*100, "\n", "process all complete !", "\n")