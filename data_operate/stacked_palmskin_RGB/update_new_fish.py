
"""

    Update new fish data

"""


import os
import sys
import re
import filecmp
from glob import glob



def create_new_dir(path:str, end="\n"):
    if not os.path.exists(path):
        # if the demo_folder directory is not exist then create it.
        os.makedirs(path)
        print(f"path: '{path}' is created!{end}")



if __name__ == "__main__":
    
    
        ap_data_root = r"C:\Users\confocal_microscope\Desktop\WorkingDir\(D2)_Image_AP\{Data}_Data"
        print(ap_data_root)
        
        
    # old path (input)

        dir_in = r"{20221109_NEW_NAME}_Academia_Sinica_i242"
        ap_data_in = os.path.join(ap_data_root, dir_in, r"stacked_palmskin_tiff")
        print(ap_data_in)
        
        ## check total of images match dir_name
        ap_data_in_list = glob(f"{ap_data_in}/*.tif*")
        total_old_img_expect = dir_in.split("i")[-1]
        assert len(ap_data_in_list) == int(total_old_img_expect), f"INPUT_FOLDER = \"{dir_in}\", but total image != {total_old_img_expect}"
        
        
    # update path (update)
    
        ap_data_update = r"C:\Users\confocal_microscope\Desktop\{NAS_DL}_Academia_Sinica_Data\{20221209_Update}_Zebrafish_A_P_strategies\stacked_palmskin_tiff\Upload_20221209"
        print(ap_data_update)
        
        ap_data_update_list = glob(f"{ap_data_update}/*.tif*")
            
    
    # new path {output}

        folder_tag = f"20221209_UPDATE_{len(ap_data_update_list)}"
        folder_postfix = len(ap_data_in_list) + len(ap_data_update_list) # total images after update
        dir_out = f"{{{folder_tag}}}_Academia_Sinica_i{folder_postfix}"
        ap_data_out =  os.path.join(ap_data_root, dir_out, r"stacked_palmskin_tiff")
        print(ap_data_out)
        
        create_new_dir(ap_data_out)
              
        
        
    # Processing
    
        print("\n\nprocessing...\n")
        
        ap_comb_list = ap_data_in_list + ap_data_update_list
        
        
        ## old path + update path --> new path
        for i, path in enumerate(ap_comb_list):
            
            
            # *** Print CMD section divider ***
            print("="*100, "\n")
            print(f"Process Count : {(i+1):{len(str(len(ap_comb_list)))}d}", "\n")
            
            
            file_name = path.split(os.sep)[-1]
            print(file_name, "\n")
            
            # Check file_name correctness
            file_name_list = re.split(" |_|-", file_name)
            # print(len(file_name_list), file_name_list)
            assert len(file_name_list) == 11, "file_name format error, expect like : '20220610_CE001_palmskin_8dpf - Series001_fish_1_A_RGB.tif'"
            
            
            new_path = os.path.join(ap_data_out, file_name)
            os.system(f"copy \"{path}\" \"{new_path}\"")
            cmp = filecmp.cmp(path, new_path) # Check correctness after copy
            
            
            # break # test 1 loop