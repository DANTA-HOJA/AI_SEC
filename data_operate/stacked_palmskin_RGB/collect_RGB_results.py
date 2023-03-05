import os
import sys
from typing import *
from datetime import datetime
import json

from glob import glob

sys.path.append(r"C:\Users\confocal_microscope\Desktop\ZebraFish_AP_POS\modules") # add path to scan customized module
from fileop import create_new_dir
from norm_name import get_fish_ID_pos, resave_result



# input
ap_data_root = r"C:\Users\confocal_microscope\Desktop\{PyIJ_OutTest}_RGB_preprocess" # r"C:\Users\confocal_microscope\Desktop\WorkingDir\(D2)_Image_AP\{Data}_Data\{20221209_UPDATE_82}_Academia_Sinica_i324"
preprocess_method_desc = "ch4_min_proj, outer_rect"
preprocess_root = os.path.join(ap_data_root, f"{{{preprocess_method_desc}}}_RGB_preprocess")


# result
result_map = {
    "B_processed":                "MetaImage/01_B_processed.tif",
	"B_processed_Kuwahara":       "MetaImage/02_B_processed_Kuwahara*.tif",
    "G_processed":                "MetaImage/03_G_processed.tif",
    "G_processed_Kuwahara":       "MetaImage/04_G_processed_Kuwahara*.tif",
    "R_processed":                "MetaImage/05_R_processed.tif",
    "R_processed_Kuwahara":       "MetaImage/06_R_processed_Kuwahara*.tif",
    "RGB":                        "MetaImage/07_composite_RGB.tif",
    "RGB_HE" :                    "08_composite_RGB_HE.tif", # CHECK_PT 
    "RGB_Kuwahara":               "MetaImage/09_composite_RGB_Kuwahara.tif",
	"RGB_Kuwahara_HE" :           "10_composite_RGB_Kuwahara_HE.tif", # CHECK_PT 
	"RGB_HE_mix" :                "11_composite_RGB_HE_mix.tif", # CHECK_PT 
    "BF_Zproj":                   "MetaImage/12_BF_Zproj_*.tif",
    "BF_Zproj_HE":                "MetaImage/13_BF_Zproj_*_HE.tif",
    "Threshold":                  "MetaImage/14_Threshold_*_*.tif",
    "outer_rect":                 "MetaImage/15_outer_rect.tif",
    "inner_rect":                 "MetaImage/16_inner_rect.tif",
    "RGB_HE--MIX":                "MetaImage/17_composite_RGB_HE--MIX.tif",
    "RGB_Kuwahara_HE--MIX":       "MetaImage/18_composite_RGB_Kuwahara_HE--MIX.tif",
    "RGB_HE_mix--MIX":            "MetaImage/19_composite_RGB_HE_mix--MIX.tif",
    "cropped_RGB_HE" :            "20_cropped_composite_RGB_HE.tif", # CHECK_PT 
	"cropped_RGB_Kuwahara_HE" :   "21_cropped_composite_RGB_Kuwahara_HE.tif", # CHECK_PT 
	"cropped_RGB_HE_mix" :        "22_cropped_composite_RGB_HE_mix.tif", # CHECK_PT 
    "RoiSet" :                    "MetaImage/RoiSet.zip",
}
result_key = "RGB_HE_mix"


# output
output_dir = os.path.join(ap_data_root, f"{{{preprocess_method_desc}}}_RGB_reCollection", result_key)
create_new_dir(output_dir)


path_list = glob(os.path.normpath("{}/*/{}".format(preprocess_root, result_map[result_key])))
path_list.sort(key=get_fish_ID_pos)
# for i in path_list: print(i)


summary = {}
summary["result_key"] = result_key
summary["actual_name"] = result_map[result_key]
summary["max_probable_num"] = get_fish_ID_pos(path_list[-1])[0]
summary["total files"] = len(path_list)
summary["missing"] = []


previous_fish = ""
for i in range(summary["max_probable_num"]):
    
    one_base_iter_num = i+1
    
    for pos in ["A", "P"]:
        
        # iter
        expect_name = f"{one_base_iter_num}_{pos}"
        # fish
        try:
            fish_ID, fish_pos = get_fish_ID_pos(path_list[0])
            current_name = f"{fish_ID}_{fish_pos}"
            assert current_name != previous_fish, f"fish_dir repeated!, check '{previous_fish}' "
        except: pass
        
        
        if one_base_iter_num == fish_ID:
            
            if fish_pos == pos :
                path = path_list.pop(0)
                resave_result(path, output_dir, result_map[result_key])
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