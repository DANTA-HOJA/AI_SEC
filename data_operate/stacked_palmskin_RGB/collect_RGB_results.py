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
    if "MetaImage" in path: fish_dir = path.split(os.sep)[-3]
    else: fish_dir = path.split(os.sep)[-2]
    fish_dir_list = re.split(" |_|-", fish_dir)
    assert len(fish_dir_list) == 11, f"len(fish_dir_list) = '{len(fish_dir_list)}', expect '11' "
    return int(fish_dir_list[8])


def get_fish_pos(path:str) -> int:
    # rgb: .../20220610_CE001_palmskin_8dpf - Series001_fish_1_A_RGB/[modality].tif
    if "MetaImage" in path: fish_dir = path.split(os.sep)[-3]
    else: fish_dir = path.split(os.sep)[-2]
    fish_dir_list = re.split(" |_|-", fish_dir)
    assert len(fish_dir_list) == 11, f"len(fish_dir_list) = '{len(fish_dir_list)}', expect '11' "
    return fish_dir_list[9]
    

def resave_RGB_result(original_path:str, output_dir:str, result_key:str):
    if "MetaImage" in path: fish_dir = path.split(os.sep)[-3]
    else: fish_dir = path.split(os.sep)[-2]
    file_ext = result_map[result_key].split(".")[-1]
    out_path = os.path.join(output_dir, f"{fish_dir}.{file_ext}")
    os.system(f"copy \"{original_path}\" \"{out_path}\" ")
    filecmp.cmp(original_path, out_path)



# input
ap_data = r"C:\Users\confocal_microscope\Desktop\{PyIJ_OutTest}_RGB_preprocess" # r"C:\Users\confocal_microscope\Desktop\WorkingDir\(D2)_Image_AP\{Data}_Data\{20221209_UPDATE_82}_Academia_Sinica_i324"
preprocess_method_desc = "ch4_min_proj, outer_rect"
preprocess_root = os.path.join(ap_data, f"{{{preprocess_method_desc}}}_RGB_preprocess")


# modality
result_map = {
    "B_processed":                "MetaImage/01_B_processed.tif",
	"B_processed_Kuwahara15":     "MetaImage/02_B_processed_Kuwahara15.tif",    # TODO:  imagej_RGB_preprocess 將 preprocess var 加入 log 後移除 para 後綴
    "G_processed":                "MetaImage/03_G_processed.tif",
    "G_processed_Kuwahara15":     "MetaImage/04_G_processed_Kuwahara15.tif",    # TODO:  imagej_RGB_preprocess 將 preprocess var 加入 log 後移除 para 後綴
    "R_processed":                "MetaImage/05_R_processed.tif",
    "R_processed_Kuwahara15":     "MetaImage/06_R_processed_Kuwahara15.tif",    # TODO:  imagej_RGB_preprocess 將 preprocess var 加入 log 後移除 para 後綴
    "RGB":                        "MetaImage/07_composite_RGB.tif",
    "RGB_HE" :                    "08_composite_RGB_HE.tif", # CHECK_PT 
    "RGB_Kuwahara":               "MetaImage/09_composite_RGB_Kuwahara.tif",
	"RGB_Kuwahara_HE" :           "10_composite_RGB_Kuwahara_HE.tif", # CHECK_PT 
	"RGB_HE_mix" :                "11_composite_HE_mix.tif", # CHECK_PT 
    "BF_Zproj_min":               "MetaImage/12_BF_Zproj_min.tif",              # TODO:  imagej_RGB_preprocess 將 preprocess var 加入 log 後移除 para 後綴
    "BF_Zproj_min_HE":            "MetaImage/13_BF_Zproj_min_HE.tif",
    "Threshold":                  "MetaImage/14_Threshold_0_10.tif",            # TODO:  imagej_RGB_preprocess 將 preprocess var 加入 log 後移除 para 後綴
    "outer_rect":                 "MetaImage/15_outer_rect.tif",
    "inner_rect":                 "MetaImage/16_inner_rect.tif",
    "RGB_HE--MIX":                "MetaImage/17_composite_RGB_HE--MIX.tif",
    "RGB_Kuwahara_HE--MIX":       "MetaImage/18_composite_RGB_Kuwahara_HE--MIX.tif",
    "RGB_HE_mix--MIX":            "MetaImage/19_composite_HE_mix--MIX.tif",
    "cropped_RGB_HE" :            "20_cropped_composite_RGB_HE.tif", # CHECK_PT 
	"cropped_RGB_Kuwahara_HE" :   "21_cropped_composite_RGB_Kuwahara_HE.tif", # CHECK_PT 
	"cropped_RGB_HE_mix" :        "22_cropped_composite_HE_mix.tif", # CHECK_PT 
    "RoiSet" :                    "MetaImage/RoiSet.zip",
}
result_key = "RGB_HE_mix"


# output
output_dir = r"C:\Users\confocal_microscope\Desktop\RGB_reCollection"
output_dir = os.path.join(output_dir, f"{{{preprocess_method_desc}}}", result_key)
create_new_dir(output_dir)


path_list = glob(os.path.normpath("{}/*/{}".format(preprocess_root, result_map[result_key])))
path_list.sort(key=get_fish_ID)
# for i in path_list: print(i)


summary = {}
summary["result_key"] = result_key
summary["actual_name"] = result_map[result_key]
summary["max_probable_file"] = get_fish_ID(path_list[-1])
summary["total files"] = len(path_list)
summary["missing"] = []


previous_fish = ""
for i in range(summary["max_probable_file"]):
    
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
                resave_RGB_result(path, output_dir, result_key)
                previous_fish = current_name
            else: # 部分缺失
                summary["missing"].append(f"{expect_name}")
                # print("missing : {}".format(summary["missing"][-1]))
            
        else: # 缺號
            summary["missing"].append(f"{expect_name}")
            # print("missing : {}".format(summary["missing"][-1]))


summary["len(missing)"] = len(summary["missing"])
print(json.dumps(summary, indent=4))
# Create log writer
time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
log_path = os.path.join(output_dir, f"{{Logs}}_{{collect_RGB_results}}_{time_stamp}.log")
log_writer = open(log_path, mode="w")
log_writer.write(json.dumps(summary, indent=4))
log_writer.close()