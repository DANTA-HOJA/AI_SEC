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
    # bf: .../20220610_CE001_palmskin_8dpf - Series001_fish_1_BF/[file_with_extension]
    if "MetaImage" in path: fish_dir = path.split(os.sep)[-3]
    else: fish_dir = path.split(os.sep)[-2]
    fish_dir_list = re.split(" |_|-", fish_dir)
    assert len(fish_dir_list) == 10, f"len(fish_dir_list) = '{len(fish_dir_list)}', expect '10' "
    return int(fish_dir_list[8])
    

def resave_BF_result(original_path:str, output_dir:str, result_key:str):
    if "MetaImage" in path: fish_dir = path.split(os.sep)[-3]
    else: fish_dir = path.split(os.sep)[-2]
    file_ext = result_map[result_key].split(".")[-1]
    out_path = os.path.join(output_dir, f"{fish_dir}.{file_ext}")
    os.system(f"copy \"{original_path}\" \"{out_path}\" ")
    filecmp.cmp(original_path, out_path)



# input
ap_data = r"C:\Users\confocal_microscope\Desktop\{PyIJ_OutTest}_BF_Analysis" # r"C:\Users\confocal_microscope\Desktop\WorkingDir\(D2)_Image_AP\{Data}_Data\{20221209_UPDATE_82}_Academia_Sinica_i324"
preprocess_method_desc = "KY_with_NameChecker"
preprocess_root = os.path.join(ap_data, f"{{{preprocess_method_desc}}}_BF_Analysis")


# result
result_map = {
	"original_16bit":      "MetaImage/01_original_16bit.tif",
	"cropped_BF" :         "02_cropped_BF.tif",
	"threshold" :          "MetaImage/03_threshold.tif",
	"measured_mask" :      "MetaImage/04_measured_mask.tif",
	"cropped_BF_mix" :     "05_cropped_BF_mix.tif",
    "RoiSet" :             "RoiSet.zip",
	"AutoAnalysis" :       "AutoAnalysis.csv",
    "ManualAnalysis" :     "ManualAnalysis.csv",
    "BothAnalysis" :       "*Analysis.csv" # both Manual and Auto will select
}
result_key = "cropped_BF_mix"


# output
output_dir = r"C:\Users\confocal_microscope\Desktop\BF_reCollection"
output_dir = os.path.join(output_dir, f"{{{preprocess_method_desc}}}_{result_key}")
create_new_dir(output_dir)


path_list = glob(os.path.normpath("{}/*/{}".format(preprocess_root, result_map[result_key])))
path_list.sort(key=get_fish_ID)
# for i in path_list: print(i)


summary = {}
summary["result_key"] = result_key
summary["actual_name"] = result_map[result_key]
summary["max_probable_fish"] = get_fish_ID(path_list[-1])
summary["lenght of modalityRGBs"] = len(path_list)
summary["missing"] = []


previous_fish = ""
for i in range(summary["max_probable_fish"]):
    
        # iter
        one_base_iter_num = i+1
        expect_name = f"{one_base_iter_num}"
        
        # fish
        try:
            fish_ID = get_fish_ID(path_list[0])
            current_name = f"{fish_ID}"
            assert current_name != previous_fish, f"fish_dir repeated!, check '{previous_fish}' "
        except: pass
        
        
        if one_base_iter_num == fish_ID:
            
            path = path_list.pop(0)
            resave_BF_result(path, output_dir, result_key)
            previous_fish = current_name
            
        else: # 缺號
            summary["missing"].append(f"{expect_name}")
            # print("missing : {}".format(summary["missing"][-1]))


summary["len(missing)"] = len(summary["missing"])
print(json.dumps(summary, indent=4))
# Create log writer
time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
log_path = os.path.join(output_dir, f"{{Logs}}_{{collect_BF_results}}_{time_stamp}.log")
log_writer = open(log_path, mode="w")
log_writer.write(json.dumps(summary, indent=4))
log_writer.close()