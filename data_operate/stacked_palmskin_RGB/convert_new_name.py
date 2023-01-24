
"""

To change image_name ( stacked_palmskin_tiff ) in new format : 

    Example of name : 

        old_name : 20220610 CE001 palmskin_8dpf - Series001 fish 1 palmskin_8dpf_A_RGB only.tif

        new_name : 20221125_AI005_palmskin_10dpf - Series001_fish_165_A_RGB.tif


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

        dir_in = r"{20221105_NAME_RP}_Academia_Sinica_i242"
        ap_data_in = os.path.join(ap_data_root, dir_in, r"stacked_palmskin_tiff")
        print(ap_data_in)
        
        ## check total of images match dir_name
        ap_data_in_list = glob(f"{ap_data_in}/*.tif*")
        total_img_expect = dir_in.split("i")[-1]
        assert len(ap_data_in_list) == int(total_img_expect), f"INPUT_FOLDER = \"{dir_in}\", but total image != {total_img_expect}"
        
        
    # rename path {output}

        dir_out = r"{20221109_NEW_NAME}_Academia_Sinica_i242"
        ap_data_out =  os.path.join(ap_data_root, dir_out, r"stacked_palmskin_RGB")
        print(ap_data_out)
        
        create_new_dir(ap_data_out)
          
        
        
    # Processing
    
        print("\n\nprocessing...\n")
        for i, path in enumerate(ap_data_in_list):
            
            
            # *** Print CMD section divider ***
            print("="*100, "\n")
            print(f"Process Count : {(i+1):{len(str(len(ap_data_in_list)))}d}", "\n")
        
        
            # old name example : 20220610 CE001 palmskin_8dpf - Series001 fish 1 palmskin_8dpf_A_RGB only.tif
            old_name = path.split(os.sep)[-1]
            old_name_list = re.split(" |_|\.", old_name)
            # print(len(old_name_list), old_name_list, "\n")
            assert len(old_name_list) == 14
        
        
            """
            Naming Plan :
            
                old name result   : ['20220610', 'CE001', 'palmskin', '8dpf', '-', 'Series001', 'fish', '1', 'palmskin', '8dpf', 'A', 'RGB', 'only', 'tif']
            
                new name example  : 20221125_AI005_palmskin_10dpf - Series001_fish_165_A_RGB.tif
            
                old name convert  : {20220610}_{CE001}_{palmskin}_{8dpf} - {Series001}_{fish}_{1}_{A}_{RGB}.tif
                        list_num  : {   0    }_{  1  }_{   2    }_{  3 } - {    5    }_{  6 }_{7}_{10}_{11}.tif
            
                generated new name : 20220610_CE001_palmskin_8dpf - Series001_fish_1_A_RGB.tif
            
            """
            gen_new_name = "{}_{}_{}_{} - {}_{}_{}_{}_{}.tif".format(old_name_list[0], old_name_list[1], old_name_list[2], 
                                                                    old_name_list[3], old_name_list[5], old_name_list[6], 
                                                                    old_name_list[7], old_name_list[10], old_name_list[11])
            print("Before : ", old_name, "\nAfter  : ", gen_new_name, "\n")
            

            new_path = os.path.join(ap_data_out, gen_new_name)
            os.system(f"copy \"{path}\" \"{new_path}\"")
            cmp = filecmp.cmp(path, new_path) # Check correctness after copy
            
            
            # break # test 1 loop