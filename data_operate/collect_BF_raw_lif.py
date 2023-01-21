
"""

    Collect all LIF_FILE in "[NAS_DL].../BrightField_raw_lif"

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
    
    
    # BrightField_raw_lif (input)
    
        bf_raw_in = r"C:\Users\confocal_microscope\Desktop\{NAS_DL}_Academia_Sinica_Data\{20221209_Update}_Zebrafish_A_P_strategies\BrightField_raw_lif"
        print(bf_raw_in, end="")
        
        bf_raw_in_list = glob(f"{bf_raw_in}/*/*.lif")
        print(f", [ found {len(bf_raw_in_list)} files ]")
    
    
    # new path {output}

        dir_out = r"{20221209_UPDATE_82}_Academia_Sinica_i324"
        bf_raw_out =  os.path.join(ap_data_root, dir_out, r"BF_Analysis")
        print(bf_raw_out)
        
        create_new_dir(bf_raw_out)
        
    
    
    # Processing
    
        print("\n\nprocessing...\n")
        
        
        ## BrightField_raw_lif --> new path
        for i, path in enumerate(bf_raw_in_list):
            
            
            # *** Print CMD section divider ***
            print("="*100, "\n")
            print(f"Process Count : {(i+1):{len(str(len(bf_raw_in_list)))}d}", "\n")
            
            
            file_name = path.split(os.sep)[-1]
            print(file_name, "\n")
            
            # Check file_name correctness
            file_name_list = re.split(" |_|-", file_name)
            # print(len(file_name_list), file_name_list)
            assert len(file_name_list) == 4, "file_name format error, expect like : '20221125_AI005_palmskin_10dpf.lif'"
            
            
            # Consociate "BrightField_raw_lif" file_name
            gen_new_name = "_".join(file_name_list)
            if gen_new_name != file_name: print(f"\ncurrent name not in expected format, generating new name ...\n\n--> {gen_new_name}\n\n")
            
            
            new_path = os.path.join(bf_raw_out, gen_new_name)
            os.system(f"copy \"{path}\" \"{new_path}\"")
            cmp = filecmp.cmp(path, new_path) # Check correctness after copy
            
            
            # break # test 1 loop       