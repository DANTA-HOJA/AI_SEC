import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Union
from datetime import datetime
import json
import toml

rel_module_path = "./../../modules/"
sys.path.append( str(Path(rel_module_path).resolve()) ) # add path to scan customized module

from fileop import create_new_dir, resave_result
from data.utils import get_fish_ID_pos



# input
ap_data_root = Path(r"/home/rime97410000/ZebraFish_DB/{Data}_Preprocessed/{20230605_rebuild_i242}_Academia_Sinica_i324")
analysis_method_desc = "KY_with_NameChecker"
analysis_root = ap_data_root.joinpath(f"{{{analysis_method_desc}}}_BrightField_analyze")
detection_mode = "missing" # missing / finding

if not ((detection_mode == "missing") or (detection_mode == "finding")): 
    raise KeyError("Key: 'detection_mode' can only be 'missing' or 'finding' ")


# result
result_map = {
	"original_16bit" :          "MetaImage/01_original_16bit.tif",
	"cropped_BF" :              "02_cropped_BF.tif", # CHECK_PT 
	"AutoThreshold" :           "MetaImage/03_AutoThreshold_*.tif",
	"measured_mask" :           "MetaImage/04_measured_mask.tif",
	"cropped_BF--MIX" :         "05_cropped_BF--MIX.tif", # CHECK_PT 
    "RoiSet" :                  "MetaImage/RoiSet.zip",
	"AutoAnalysis" :            "AutoAnalysis.csv",
    "ManualAnalysis" :          "ManualAnalysis.csv",
    "Manual_measured_mask" :    "Manual_measured_mask.tif", # CHECK_PT 
    "Manual_cropped_BF--MIX" :  "Manual_cropped_BF--MIX.tif", # CHECK_PT 
}
result_key = "ManualAnalysis"


# output
output_dir = ap_data_root.joinpath(f"{{{analysis_method_desc}}}_BrightField_reCollection", result_key)
create_new_dir(output_dir)


path_list = sorted(analysis_root.glob(f"*/{result_map[result_key]}"), key=get_fish_ID_pos)
# for i in path_list: print(i)


summary = {}
summary["result_key"] = result_key
summary["actual_name"] = result_map[result_key]
summary["max_probable_num"] = get_fish_ID_pos(path_list[-1])[0]
summary["total files"] = len(path_list)
summary[detection_mode] = []


previous_fish = ""
for i in range(summary["max_probable_num"]):
    
        # iter
        one_base_iter_num = i+1
        expect_name = f"{one_base_iter_num}"
        
        # fish
        try:
            fish_ID, _ = get_fish_ID_pos(path_list[0])
            current_name = f"{fish_ID}"
            assert current_name != previous_fish, f"fish_dir repeated!, check '{previous_fish}' "
        except: pass
        
        
        if current_name == expect_name:
            
            path = path_list.pop(0)
            resave_result(path, output_dir, result_map[result_key])
            previous_fish = current_name
            
            if detection_mode == "finding": summary[detection_mode].append(f"{expect_name}")
        
        else: 
            if detection_mode == "missing": summary[detection_mode].append(f"{expect_name}")
            
        # print(f"{detection_mode} : {summary[detection_mode]}")


summary[f"len({detection_mode})"] = len(summary[detection_mode])
print(json.dumps(summary, indent=4))
# Create log writer
time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
log_path = os.path.join(output_dir, f"{{Logs}}_{{collect_BF_results}}_{time_stamp}.log")
log_writer = open(log_path, mode="w")
log_writer.write(json.dumps(summary, indent=4))
log_writer.close()