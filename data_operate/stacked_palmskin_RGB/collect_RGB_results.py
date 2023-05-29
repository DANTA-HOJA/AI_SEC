import os
import sys
from typing import *
from datetime import datetime
import json

from glob import glob

sys.path.append("./../../modules/") # add path to scan customized module
from fileop import create_new_dir, resave_result
from data.utils import get_fish_ID_pos



# input
ap_data_root = r"C:\Users\confocal_microscope\Desktop\{Temp}_Data\{20230424_Update}_Academia_Sinica_i505"
preprocess_method_desc = "ch4_min_proj, outer_rect"
preprocess_root = os.path.join(ap_data_root, f"{{{preprocess_method_desc}}}_RGB_preprocess")


# result
result_map = {
    "RGB_direct_max_zproj":          "*_RGB_direct_max_zproj.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "ch_B":                          "MetaImage/*_B_processed.tif",
	"ch_B_Kuwahara":                 "MetaImage/*_B_processed_Kuwahara*.tif",
    "ch_B_fusion":                   "*_B_processed_fusion.tif", # CHECK_PT 
    "ch_B_HE":                       "MetaImage/*_B_processed_HE.tif",
    "ch_B_Kuwahara_HE":              "MetaImage/*_B_processed_Kuwahara_HE.tif",
    "ch_B_fusion":                   "*_B_processed_HE_fusion.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "ch_G":                          "MetaImage/*_G_processed.tif",
    "ch_G_Kuwahara":                 "MetaImage/*_G_processed_Kuwahara*.tif",
    "ch_G_fusion":                   "*_G_processed_fusion.tif", # CHECK_PT 
    "ch_G_HE":                       "MetaImage/*_G_processed_HE.tif",
    "ch_G_Kuwahara_HE":              "MetaImage/*_G_processed_Kuwahara_HE.tif",
    "ch_G_fusion":                   "*_G_processed_HE_fusion.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "ch_R":                          "MetaImage/*_R_processed.tif",
    "ch_R_Kuwahara":                 "MetaImage/*_R_processed_Kuwahara*.tif",
    "ch_R_fusion":                   "*_R_processed_fusion.tif", # CHECK_PT 
    "ch_R_HE":                       "MetaImage/*_R_processed_HE.tif",
    "ch_R_Kuwahara_HE":              "MetaImage/*_R_processed_Kuwahara_HE.tif",
    "ch_R_fusion":                   "*_R_processed_HE_fusion.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "RGB":                           "MetaImage/*_RGB_processed.tif",
    "RGB_Kuwahara":                  "MetaImage/*_RGB_processed_Kuwahara.tif",
    "RGB_fusion":                    "*_RGB_processed_fusion.tif", # CHECK_PT  = Average(RGB_processed, RGB_processed_Kuwahara)
    "RGB_HE" :                       "MetaImage/*_RGB_processed_HE.tif",
	"RGB_Kuwahara_HE" :              "MetaImage/*_RGB_processed_Kuwahara_HE.tif",
	"RGB_HE_fusion" :                "*_RGB_processed_HE_fusion.tif", # CHECK_PT  = Average(RGB_processed_HE, RGB_processed_Kuwahara_HE)
    # -----------------------------------------------------------------------------------
    "BF_Zproj":                      "MetaImage/*_BF_Zproj_*.tif",
    "BF_Zproj_HE":                   "MetaImage/*_BF_Zproj_*_HE.tif",
    "Threshold":                     "MetaImage/*_Threshold_*_*.tif",
    "outer_rect":                    "MetaImage/*_outer_rect.tif",
    "inner_rect":                    "MetaImage/*_inner_rect.tif",
    "RoiSet" :                       "MetaImage/RoiSet_AutoRect.zip",
    # -----------------------------------------------------------------------------------
    "RGB_Kuwahara_HE--AutoRect":     "*_RGB_processed_fusion--AutoRect.tif", # CHECK_PT 
    "RGB_HE_fusion--AutoRect":       "*_RGB_processed_HE_fusion--AutoRect.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "autocropped_RGB_fusion" :       "*_autocropped_RGB_processed_fusion.tif", # CHECK_PT 
	"autocropped_RGB_HE_fusion" :    "*_autocropped_RGB_processed_HE_fusion.tif", # CHECK_PT 
}
result_key = "RGB_HE_fusion"


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