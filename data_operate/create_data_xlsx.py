
"""

To generate data information in XLSX ( XLSX file will used to compute the classes in classification task ):

    All fish will process with the following step : 
    
        1. Run ImageJ Macro : Use bright field (BF) images to compute the surface area (SA) and surface length (SL), and store their results in CSV format.
        2. Collect all generated CSV files using pandas.DataFrame().
        3. Use "fish_id" to find and add their "palmskin_RGB" images into the DataFrame.
        4. Save results in XLSX format.

"""


import os
import sys
import re
from typing import *
from glob import glob

import numpy as np
import pandas as pd

sys.path.append(r"C:\Users\confocal_microscope\Desktop\ZebraFish_AP_POS\modules")
from logger import init_logger
from dataop import get_fish_ID_pos, create_dict_by_fishID, merge_BF_analysis



log = init_logger(r"Create_XLSX")
# log.debug('debug message')
# log.info('info message')
# log.warning('warning message')
# log.error('error message')
# log.critical('critical message')



if __name__ == "__main__":


            # user variable
            ap_data_root = r"C:\Users\confocal_microscope\Desktop\{Temp}_Data\{20230424_Update}_Academia_Sinica_i505"
            print(ap_data_root)
            # --- BF
            analysis_method_desc = "KY_with_NameChecker"
            # --- RGB
            preprocess_method_desc = "ch4_min_proj, outer_rect"
            RGB_recollect_key = "RGB_HE_mix"
            # --- use to remove
            bf_bad_condition = [] # 4, 7, 68, 109, 156
            delete_uncomplete_row = True


    # BF_Analysis (input)
    
        # Grabbing files
            bf_recollect_root = os.path.join(ap_data_root, f"{{{analysis_method_desc}}}_BF_reCollection")
            bf_recollect_auto = os.path.join(bf_recollect_root, "AutoAnalysis")
            bf_recollect_manual = os.path.join(bf_recollect_root, "ManualAnalysis")
            bf_recollect_auto_list = glob(os.path.normpath(f"{bf_recollect_auto}/*.csv"))
            bf_recollect_manual_list = glob(os.path.normpath(f"{bf_recollect_manual}/*.csv"))
        # Check grabbing error: List Empty
            assert len(bf_recollect_auto_list) > 0, "Can't find `BF_reCollection/AutoAnalysis` folder, or it is empty."
        # Do sort because the os grabbing strategy ( for example, 10 will list before 8 )
            bf_recollect_auto_list.sort(key=get_fish_ID_pos)
            bf_recollect_manual_list.sort(key=get_fish_ID_pos)
            log.info(f"Found {len(bf_recollect_auto_list)} AutoAnalysis.csv, {len(bf_recollect_manual_list)} ManualAnalysis.csv, Total: {len(bf_recollect_auto_list) + len(bf_recollect_manual_list)} files")
        # Merging `AutoAnalysis` and `ManualAnalysis` list
            bf_recollect_auto_dict = create_dict_by_fishID(bf_recollect_auto_list)
            bf_recollect_manual_dict = create_dict_by_fishID(bf_recollect_manual_list)
            bf_recollect_merge_dict = merge_BF_analysis(bf_recollect_auto_dict, bf_recollect_manual_dict)
            bf_recollect_merge_list = list(bf_recollect_merge_dict.values())
            bf_recollect_merge_list.sort(key=get_fish_ID_pos)
            log.info(f"After Merging , Total: {len(bf_recollect_merge_list)} files")
            # for i, path in enumerate(bf_recollect_merge_list): log.info(f'{path.split(os.sep)[-2]}, path {type(path)}: SN:{i:{len(str(len(bf_recollect_merge_list)))}}, {path.split(os.sep)[-1]}')

        
    # stacked_palmskin_RGB (input)
    
        # Grabbing files
            RGB_recollect_root = os.path.join(ap_data_root, f"{{{preprocess_method_desc}}}_RGB_reCollection")
            RGB_recollect_type = os.path.join(RGB_recollect_root, RGB_recollect_key)
            RGB_recollect_type_list = glob(os.path.normpath(f"{RGB_recollect_type}/*.tif"))
        # Check grabbing error: List Empty
            assert len(RGB_recollect_type_list) > 0, f"Can't find 'RGB_reCollection/{RGB_recollect_key}' folder, or it is empty."
        # Do sort because the os grabbing strategy ( for example, 10 will list before 8 )
            RGB_recollect_type_list.sort(key=get_fish_ID_pos)
            log.info(f"Found {len(RGB_recollect_type_list)} RGB tif files")
            # for i, path in enumerate(RGB_recollect_type_list): log.info(f'{RGB_recollect_key}, path {type(path)}: SN:{i:{len(str(len(RGB_recollect_type_list)))}}, {path.split(os.sep)[-1]}')
        
        
    # data.xlsx (output)
    
            output = os.path.join(ap_data_root, r"data.xlsx")
        
        
    # Processing
    
            print("\n\nprocessing...\n")

            # Creating "data.xlsx"
            data = pd.DataFrame(columns=["BrightField name with Analysis statement (CSV)",
                                         "Anterior (SP8, .tif)", 
                                         "Posterior (SP8, .tif)",
                                         "Trunk surface area, SA (um2)",
                                         "Standard Length, SL (um)"])
            
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
                    bf_result_path_list = bf_result_path.split(os.sep)
                    bf_result_name = bf_result_path_list[-1].split(".")[0] # Get name_noExtension
                    bf_result_analysis_type = bf_result_path_list[-2] # `AutoAnalysis` or `ManualAnalysis`
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
                
                
                if f"{one_base_iter_num}_A" in RGB_recollect_type_list[0] :
                    palmskin_RGB_A_name = RGB_recollect_type_list.pop(0).split(os.sep)[-1].split(".")[0] # Get name_noExtension
                    log.info(f'palmskin_RGB_A_name {type(palmskin_RGB_A_name)}: {palmskin_RGB_A_name}')
                    data.loc[one_base_iter_num, "Anterior (SP8, .tif)" ] =  palmskin_RGB_A_name
                
                
                if f"{one_base_iter_num}_P" in RGB_recollect_type_list[0] :
                    palmskin_RGB_P_name = RGB_recollect_type_list.pop(0).split(os.sep)[-1].split(".")[0] # Get name_noExtension
                    log.info(f'palmskin_RGB_P_name {type(palmskin_RGB_P_name)}: {palmskin_RGB_P_name}')
                    data.loc[one_base_iter_num, "Posterior (SP8, .tif)" ] =  palmskin_RGB_P_name
                
                
                print("\n\n\n")
            
            
            if delete_uncomplete_row: data.dropna(inplace=True)
            if bf_bad_condition: data:pd.DataFrame = data[data.index.isin(bf_bad_condition) == False]
            data.to_excel(output, engine="openpyxl")
    
            print("="*100, "\n", "process all complete !", "\n")