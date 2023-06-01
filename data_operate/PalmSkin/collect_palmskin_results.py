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


# -----------------------------------------------------------------------------------
# Load `db_path_plan.toml`
with open("./../../Config/db_path_plan.toml", mode="r") as f_reader:
    dbpp_config = toml.load(f_reader)
db_root = Path(dbpp_config["root"])

# -----------------------------------------------------------------------------------
# Load `(CollectResult)_palmskin.toml`
with open("./../../Config/(CollectResult)_palmskin.toml", mode="r") as f_reader:
    config = toml.load(f_reader)
preprocessed_desc = config["data_preprocessed"]["desc"]
result_alias = config["result_alias"]

# -----------------------------------------------------------------------------------
# Generate `path_vars`

# Check `{desc}_Academia_Sinica_i[num]`
data_preprocessed_root = db_root.joinpath(dbpp_config["data_preprocessed"])
target_dir_list = list(data_preprocessed_root.glob(f"*{preprocessed_desc}*"))
assert len(target_dir_list) == 1, (f"[data_preprocessed.desc] in `(CollectResult)_palmskin.toml` is not unique/exists, "
                                   f"find {len(target_dir_list)} possible directories, {target_dir_list}")
data_preprocessed_dir = target_dir_list[0]

# Check `{reminder}_PalmSkin_preprocess`
target_dir_list = list(data_preprocessed_dir.glob(f"*PalmSkin_preprocess*"))
assert len(target_dir_list) == 1, (f"found {len(target_dir_list)} directories, only one `PalmSkin_preprocess` is accepted.")
palmskin_preprocess_dir = target_dir_list[0]
palmskin_preprocessed_reminder = re.split("{|}", str(palmskin_preprocess_dir).split(os.sep)[-1])[1]


# -----------------------------------------------------------------------------------
# Load `palmskin_preprocess_config.toml`
palmskin_preprocess_config_path = palmskin_preprocess_dir.joinpath("palmskin_preprocess_config.toml")

with open(palmskin_preprocess_config_path, mode="r") as f_reader:
    palmskin_preprocess_config = toml.load(f_reader)

preprocess_kwargs = palmskin_preprocess_config["param"]
Kuwahara = f"Kuwahara{preprocess_kwargs['Kuwahara_sampleing']}"
bf_zproj_type = f"BF_Zproj_{preprocess_kwargs['bf_zproj_type']}"
bf_treshold = f"0_{preprocess_kwargs['bf_treshold_value']}"



# -----------------------------------------------------------------------------------
result_map = {
    "RGB_direct_max_zproj":          "*_RGB_direct_max_zproj.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "ch_B":                          "MetaImage/*_B_processed.tif",
	"ch_B_Kuwahara":                 f"MetaImage/*_B_processed_{Kuwahara}.tif",
    "ch_B_fusion":                   "*_B_processed_fusion.tif", # CHECK_PT 
    "ch_B_HE":                       "MetaImage/*_B_processed_HE.tif",
    "ch_B_Kuwahara_HE":              f"MetaImage/*_B_processed_{Kuwahara}_HE.tif",
    "ch_B_fusion":                   "*_B_processed_HE_fusion.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "ch_G":                          "MetaImage/*_G_processed.tif",
    "ch_G_Kuwahara":                 f"MetaImage/*_G_processed_{Kuwahara}.tif",
    "ch_G_fusion":                   "*_G_processed_fusion.tif", # CHECK_PT 
    "ch_G_HE":                       "MetaImage/*_G_processed_HE.tif",
    "ch_G_Kuwahara_HE":              f"MetaImage/*_G_processed_{Kuwahara}_HE.tif",
    "ch_G_fusion":                   "*_G_processed_HE_fusion.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "ch_R":                          "MetaImage/*_R_processed.tif",
    "ch_R_Kuwahara":                 f"MetaImage/*_R_processed_{Kuwahara}.tif",
    "ch_R_fusion":                   "*_R_processed_fusion.tif", # CHECK_PT 
    "ch_R_HE":                       "MetaImage/*_R_processed_HE.tif",
    "ch_R_Kuwahara_HE":              f"MetaImage/*_R_processed_{Kuwahara}_HE.tif",
    "ch_R_fusion":                   "*_R_processed_HE_fusion.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "RGB":                           "MetaImage/*_RGB_processed.tif",
    "RGB_Kuwahara":                  f"MetaImage/*_RGB_processed_{Kuwahara}.tif",
    "RGB_fusion":                    "*_RGB_processed_fusion.tif", # CHECK_PT  = Average(RGB_processed, RGB_processed_Kuwahara)
    "RGB_fusion2Gray":               "*_RGB_processed_fusion2Gray.tif", # CHECK_PT 
    "RGB_HE" :                       "MetaImage/*_RGB_processed_HE.tif",
	"RGB_Kuwahara_HE" :              f"MetaImage/*_RGB_processed_{Kuwahara}_HE.tif",
	"RGB_HE_fusion" :                "*_RGB_processed_HE_fusion.tif", # CHECK_PT  = Average(RGB_processed_HE, RGB_processed_Kuwahara_HE)
    "RGB_HE_fusion2Gray":            "*_RGB_processed_HE_fusion2Gray.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "BF_Zproj":                      f"MetaImage/*_{bf_zproj_type}.tif",
    "BF_Zproj_HE":                   f"MetaImage/*_{bf_zproj_type}_HE.tif",
    "Threshold":                     f"MetaImage/*_Threshold_{bf_treshold}.tif",
    "outer_rect":                    "MetaImage/*_outer_rect.tif",
    "inner_rect":                    "MetaImage/*_inner_rect.tif",
    "RoiSet" :                       "MetaImage/RoiSet_AutoRect.zip",
    # -----------------------------------------------------------------------------------
    "RGB_fusion--AutoRect":          "*_RGB_processed_fusion--AutoRect.tif", # CHECK_PT 
    "RGB_HE_fusion--AutoRect":       "*_RGB_processed_HE_fusion--AutoRect.tif", # CHECK_PT 
    # -----------------------------------------------------------------------------------
    "autocropped_RGB_fusion" :       "*_autocropped_RGB_processed_fusion.tif", # CHECK_PT 
	"autocropped_RGB_HE_fusion" :    "*_autocropped_RGB_processed_HE_fusion.tif", # CHECK_PT 
}


# output
output_dir = data_preprocessed_dir.joinpath(f"{{{palmskin_preprocessed_reminder}}}_PalmSkin_reCollection", result_alias)
assert not output_dir.exists(), f"Directory: '{output_dir}' already exists, please delete the folder before collecting results."
create_new_dir(output_dir)


# regex filter
path_list = sorted(palmskin_preprocess_dir.glob(f"*/{result_map[result_alias]}"), key=get_fish_ID_pos)
pattern = result_map[result_alias].split(os.sep)[-1]
pattern = pattern.replace("*", r"[0-9]*")
num = 0
actual_name = None
for _ in range(len(path_list)):
    path_str = str(path_list[num]).split(os.sep)[-1]
    if not re.fullmatch(pattern, path_str):
        path_list.pop(num)
    else: 
        num += 1
        if actual_name is None: actual_name = path_str
# for i in path_list: print(i)


summary = {}
summary["result_alias"] = result_alias
summary["actual_name"] = actual_name
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
                resave_result(path, output_dir, result_map[result_alias])
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


# rename dir
new_name = f"{{{preprocessed_desc}}}_Academia_Sinica_i{summary['total files']}"
os.rename(data_preprocessed_dir, data_preprocessed_root.joinpath(new_name))