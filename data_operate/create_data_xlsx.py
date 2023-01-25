
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
import filecmp
from glob import glob
import logging

import pandas as pd

log: logging.Logger = logging.getLogger(name='dev')
log.setLevel(logging.DEBUG)

handler: logging.StreamHandler = logging.StreamHandler()
formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

# log.debug('debug message')
# log.info('info message')
# log.warning('warning message')
# log.error('error message')
# log.critical('critical message')



def get_fish_ID(path:str) -> int:
    
    """To get fish ID

    Args:
        path (str): file path

    Returns:
        int: fish id
    """
    
    fileName_NoExtension = path.split(os.sep)[-1].split(".")[0]
    split_into_2_part = fileName_NoExtension.split("fish")
    assert len(split_into_2_part) == 2, "ERROR: fileName_NoExtension.split()"
    
    post_part = split_into_2_part[1].replace(" ", "_") # generally looks like: _1_AutoAnalysis, _11_palmskin_8dpf_AutoAnalysis, _172_BF_AutoAnalysis, 
                                                       # but fish_179 not in format (confocal naming error): 179_BF-delete_AutoAnalysis, 
                                                       # so we need to deal with it.
    
    post_part_split_list = post_part.split("_")
    if "" in post_part_split_list: post_part_split_list.remove('')
    
    fish_id = int(post_part_split_list[0])
    # log.info(f'fish_id {type(fish_id)}: {fish_id}')
    
    return fish_id



if __name__ == "__main__":
    
            ap_data_root = r"C:\Users\confocal_microscope\Desktop\WorkingDir\(D2)_Image_AP\{Data}_Data\{20221209_UPDATE_82}_Academia_Sinica_i324"
            print(ap_data_root)
    
    # BF_Analysis (input)
    
        # Grabbing files
            bf_result_in = os.path.join(ap_data_root, r"BF_Analysis", "BF_Analysis--Result")
            bf_result_in_list = glob(f"{bf_result_in}/*.csv")
        # Check grabbing error: List Empty
            assert len(bf_result_in_list) > 0, "Can't find 'BF_Analysis' folder, or it is empty."
        # Do sort because the os grabbing strategy ( for example, 10 will list before 8 )
            bf_result_in_list.sort(key=get_fish_ID)
            for i, path in enumerate(bf_result_in_list): log.info(f'path {type(path)}: SN:{i}, {path.split(os.sep)[-1]}')

        
    # stacked_palmskin_RGB (input)
    
        # Grabbing files
            palmskin_RGB_in = os.path.join(ap_data_root, r"stacked_palmskin_RGB")
            palmskin_RGB_in_list = glob(f"{palmskin_RGB_in}/*.tif*")
        # Check grabbing error: List Empty
            assert len(palmskin_RGB_in_list) > 0, "Can't find 'stacked_palmskin_RGB' folder, or it is empty."
        # Do sort because the os grabbing strategy ( for example, 10 will list before 8 )
            palmskin_RGB_in_list.sort(key=get_fish_ID)
            for i, path in enumerate(palmskin_RGB_in_list): log.info(f'path {type(path)}: SN:{i}, {path.split(os.sep)[-1]}')
        
        
    # data.xlsx (output)
    
            output = os.path.join(ap_data_root, r"data.xlsx")
        
        
    # # Processing
    
            print("\n\nprocessing...\n")

            # Creating "data.xlsx"
            data = pd.DataFrame(columns=["BrightField name with Analysis statement (CSV)",
                                         "Anterior (SP8, .tif)", 
                                         "Posterior (SP8, .tif)",
                                         "Trunk surface area, SA (um2)",
                                         "Standard Length, SL (um)"])
            
            # Variable
            max_probable_fish = get_fish_ID(bf_result_in_list[-1])
            log.info(f'max_probable_fish {type(max_probable_fish)}: {max_probable_fish}\n')
            bf_result_iter_i = 0
            palmskin_RGB_iter_i = 0
            
            
            # Starting...
            for i in range(max_probable_fish):
                
                # *** Print CMD section divider ***
                print("="*100, "\n")
                
                one_base_iter_num = i+1 # Make iteration starting number start from 1
                log.info(f'one_base_iter_num {type(one_base_iter_num)}: {one_base_iter_num}\n')
                
                
                bf_result_name = bf_result_in_list[0].split(os.sep)[-1].split(".")[0] # # Get name_noExtension
                if  one_base_iter_num == get_fish_ID(bf_result_name) :
                    
                    log.info(f'bf_result_name {type(bf_result_name)}: {bf_result_name}')
                    # Read CSV
                    analysis_csv = pd.read_csv(bf_result_in_list.pop(0), index_col=" ")
                    # Get surface area from analysis file
                    surface_area = analysis_csv.loc[1, "Area"]
                    log.info(f'surface_area {type(surface_area)}: {surface_area}')
                    # Get standard length from analysis file
                    standard_length = analysis_csv.loc[1, "Feret"]
                    log.info(f'standard_length {type(standard_length)}: {standard_length}')
                    
                    data.loc[one_base_iter_num, "BrightField name with Analysis statement (CSV)"] = bf_result_name
                    data.loc[one_base_iter_num, "Trunk surface area, SA (um2)"] = surface_area
                    data.loc[one_base_iter_num, "Standard Length, SL (um)"] = standard_length

                else: data.loc[one_base_iter_num] = "" # Can't find corresponding analysis result, make an empty row.
                
                
                if f"{one_base_iter_num}_A" in palmskin_RGB_in_list[0] :
                    palmskin_RGB_A_name = palmskin_RGB_in_list.pop(0).split(os.sep)[-1].split(".")[0] # Get name_noExtension
                    log.info(f'palmskin_RGB_A_name {type(palmskin_RGB_A_name)}: {palmskin_RGB_A_name}')
                    data.loc[one_base_iter_num, "Anterior (SP8, .tif)" ] =  palmskin_RGB_A_name
                
                
                if f"{one_base_iter_num}_P" in palmskin_RGB_in_list[0] :
                    palmskin_RGB_P_name = palmskin_RGB_in_list.pop(0).split(os.sep)[-1].split(".")[0] # Get name_noExtension
                    log.info(f'palmskin_RGB_P_name {type(palmskin_RGB_P_name)}: {palmskin_RGB_P_name}')
                    data.loc[one_base_iter_num, "Posterior (SP8, .tif)" ] =  palmskin_RGB_P_name
                
                
                print("\n\n\n")
            
            data.to_excel(output, engine="openpyxl")
    
            print("="*100, "\n", "process all complete !", "\n")