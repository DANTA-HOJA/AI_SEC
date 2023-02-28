import os
import sys
import re
from typing import *
from datetime import datetime
import filecmp
import json

from glob import glob

sys.path.append(r"C:\Users\confocal_microscope\Desktop\ZebraFish_AP_POS\modules") # add path to scan customized module
from fileop import create_new_dir


def get_fish_ID(path:str) -> int:
    # rgb: .../20220610_CE001_palmskin_8dpf - Series001_fish_1_A_RGB/[modality].tif
    fish_dir = path.split(os.sep)[-2]
    fish_dir_list = re.split(" |_|-", fish_dir)
    assert len(fish_dir_list) == 11, f"len(fish_dir_list) = '{len(fish_dir_list)}', expect '11' "
    return int(fish_dir_list[8])


def get_fish_pos(path:str) -> int:
    # rgb: .../20220610_CE001_palmskin_8dpf - Series001_fish_1_A_RGB/[modality].tif
    fish_dir = path.split(os.sep)[-2]
    fish_dir_list = re.split(" |_|-", fish_dir)
    assert len(fish_dir_list) == 11, f"len(fish_dir_list) = '{len(fish_dir_list)}', expect '11' "
    return fish_dir_list[9]
    

def resave_modality_RGB(original_path:str, output_dir:str):
    fish_dir = original_path.split(os.sep)[-2]
    out_path = os.path.join(output_dir, f"{fish_dir}.tif")
    os.system(f"copy \"{original_path}\" \"{out_path}\" ")
    filecmp.cmp(original_path, out_path)



# input
ap_data = r"C:\Users\confocal_microscope\Desktop\{PyIJ_OutTest}_RGB_preprocess" # r"C:\Users\confocal_microscope\Desktop\WorkingDir\(D2)_Image_AP\{Data}_Data\{20221209_UPDATE_82}_Academia_Sinica_i324"
preprocess_method_desc = "ch4_min_proj, outer_rect"
preprocess_root = os.path.join(ap_data, f"{{{preprocess_method_desc}}}_RGB_preprocess")


# modality
modality_map = {
	"RGB_HE":                   "08_composite_RGB_HE.tif",
	"RGB_Kuwahara_HE" :         "10_composite_RGB_Kuwahara_HE.tif",
	"RGB_Mix" :                 "11_composite_HE_mix.tif",
	"cropped_RGB_HE" :          "20_cropped_composite_RGB_HE.tif",
	"cropped_RGB_Kuwahara_HE" : "21_cropped_composite_RGB_Kuwahara_HE.tif",
	"cropped_RGB_Mix" :         "22_cropped_composite_HE_mix.tif"
}
modality = "RGB_Mix"


# output
output_dir = r"C:\Users\confocal_microscope\Desktop\RGB_modality"
output_dir = os.path.join(output_dir, f"{{{preprocess_method_desc}}}_{modality}")
create_new_dir(output_dir)


path_list = glob(os.path.normpath("{}/*/{}".format(preprocess_root, modality_map[modality])))
path_list.sort(key=get_fish_ID)
# for i in path_list: print(i)


summary = {}
summary["modality"] = modality
summary["preprocessed_name"] = modality_map[modality]
summary["max_probable_fish"] = get_fish_ID(path_list[-1])
summary["lenght of modalityRGBs"] = len(path_list)
summary["missing"] = []


previous_fish = ""
for i in range(summary["max_probable_fish"]):
    
    one_base_iter_num = i+1
    
    for pos in ["A", "P"]:
        
        # iter
        expect_name = f"{one_base_iter_num}_{pos}"
        # fish
        try:
            fish_ID = get_fish_ID(path_list[0])
            fish_pos = get_fish_pos(path_list[0])
            current_name = f"{fish_ID}_{fish_pos}"
            assert current_name != previous_fish, f"fish_dir repeated!, check '{previous_fish}' "
        except: pass
        
        
        if one_base_iter_num == fish_ID:
            
            if fish_pos == pos :
                path = path_list.pop(0)
                resave_modality_RGB(path, output_dir)
                previous_fish = current_name
            else: # 部分缺失
                summary["missing"].append(f"{expect_name}")
                # print("missing : {}".format(summary["missing"][-1]))
            
        else: # 缺號
            summary["missing"].append(f"{expect_name}")
            # print("missing : {}".format(summary["missing"][-1]))



print(json.dumps(summary, indent=4))
# Create log writer
time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
log_path = os.path.join(output_dir, f"{{Logs}}_{{collect_RGBs_modality}}_{time_stamp}.log")
log_writer = open(log_path, mode="w")
log_writer.write(json.dumps(summary, indent=4))
log_writer.close()