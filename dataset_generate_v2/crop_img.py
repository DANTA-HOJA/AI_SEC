import os
import sys
import shutil
import re
from colorama import Fore, Back, Style
from pathlib import Path
import toml

from tqdm.auto import tqdm
import numpy as np
import cv2

abs_module_path = Path("./../modules/").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module
    
from fileop import create_new_dir
from data.utils import get_fish_id_pos
from data.ProcessedDataInstance import ProcessedDataInstance
from dataset.utils import gen_crop_img, sort_fish_dsname

config_dir = Path( "./../Config/" ).resolve()

# -----------------------------------------------------------------------------------
config_name = "(MakeDataset)_horiz_cut.toml"

with open(config_dir.joinpath(config_name), mode="r") as f_reader:
    config = toml.load(f_reader)

script_name = config["script_name"]
processed_inst_desc   = config["data_processed"]["desc"]
clustered_xlsx_file   = config["data_processed"]["clustered_xlsx_file"]
palmskin_result_alias = config["data_processed"]["palmskin_result_alias"]
crop_size    = config["gen_param"]["crop_size"]
shift_region = config["gen_param"]["shift_region"]
intensity    = config["gen_param"]["intensity"]
drop_ratio   = config["gen_param"]["drop_ratio"]
random_seed  = config["gen_param"]["random_seed"]
random_state = np.random.RandomState(seed=random_seed)

# -----------------------------------------------------------------------------------
# Initialize a `ProcessedDataInstance` object

processed_data_instance = ProcessedDataInstance(config_dir, processed_inst_desc)
# check images are existing and readable
# relative_path_in_fish_dir = processed_data_instance.check_palmskin_images_condition(palmskin_result_alias)

# -----------------------------------------------------------------------------------
# Generate `path_vars`

# dataset_dir
instance_name = processed_data_instance.instance_name
dataset_root = processed_data_instance.db_root.joinpath(processed_data_instance.dbpp_config["dataset_cropped_v2"])

save_dir_A_only = dataset_root.joinpath(instance_name, palmskin_result_alias, "fish_dataset_horiz_cut_1l2_A_only")
save_dir_P_only = dataset_root.joinpath(instance_name, palmskin_result_alias, "fish_dataset_horiz_cut_1l2_P_only")
save_dir_Mix_AP = dataset_root.joinpath(instance_name, palmskin_result_alias, "fish_dataset_horiz_cut_1l2_Mix_AP")

# check directory existence
not_found_cnt = 0
for dir in [save_dir_A_only, save_dir_P_only, save_dir_Mix_AP]:
    if not os.path.exists(dir): 
        print(f"{Fore.RED}{Back.BLACK} Can't find directory: '{dir}' {Style.RESET_ALL}")
        not_found_cnt += 1
assert not_found_cnt == 0, f"{Fore.RED} Can't find directories, run `mk_dataset_horiz_cut.py` before crop. {Style.RESET_ALL}\n"

# -----------------------------------------------------------------------------------
# Detect `CropSize_{crop_size}` directories

replace = True #  TODO:  可以在 config 多加一個 replace 參數，選擇要不要重切

existing_crop_dir = []
for dir in [save_dir_A_only, save_dir_P_only, save_dir_Mix_AP]:
    temp_list = list(dir.glob(f"*/*/CropSize_{crop_size}"))
    print(f"{Fore.YELLOW}{Back.BLACK} Detect {len(temp_list)} 'CropSize_{crop_size}' directories in '{dir}' {Style.RESET_ALL}")
    existing_crop_dir.extend(temp_list)

if existing_crop_dir:
    if replace: # (config varname TBA)
        print(f"Deleting {len(existing_crop_dir)} 'CropSize_{crop_size}' directories")
        for dir in existing_crop_dir: shutil.rmtree(dir)
        print("Done!")
    else:
        raise FileExistsError(f"{Fore.YELLOW}{Back.BLACK} To re-crop the images, set `config.replace` = True {Style.RESET_ALL}")

# -----------------------------------------------------------------------------------
# Do `Crop`
# WARNING: 不要使用 glob 的 ** 語法 scan tiff，否則會撈到其他 cropped images 形成無窮巢狀結構

img_paths = sorted(save_dir_Mix_AP.glob(f"*/*/*.tiff"), key=sort_fish_dsname)
pbar = tqdm(total=len(img_paths), desc=f"[ Crop Images ] : ")

for path in img_paths:
    
    img = cv2.imread(str(path))
    
    # extract info
    path_split = str(path).split(os.sep)
    fish_dsname = path_split[-2]
    file_ext = path_split[-1].split(".")[-1]
    
    # generate `save_dir`
    save_dir = path_split[:-1]
    save_dir.append(f"CropSize_{crop_size}")
    save_dir = Path(os.sep.join(save_dir))
    create_new_dir(save_dir, display_in_CLI=False)
    
    # cropping
    crop_img_list = gen_crop_img(img, crop_size, shift_region)
    
    if pbar.total != len(img_paths)*len(crop_img_list):
        pbar.total = len(img_paths)*len(crop_img_list)
    
    for i, cropped_img in enumerate(crop_img_list):
        
        cropped_name = f"{fish_dsname}_crop_{i}"
        
        pbar.desc = f"[ Crop Images ] {cropped_name} : "
        pbar.refresh()
        
        save_path = save_dir.joinpath(f"{cropped_name}.{file_ext}")
        cv2.imwrite(str(save_path), cropped_img)
        
        pbar.update(1)
        pbar.refresh()

pbar.close()

# -----------------------------------------------------------------------------------
# save_dir_Mix_AP ---copy---> save_dir_{pos}_only

for pos in ["A", "P"]:
    
    img_paths = sorted(save_dir_Mix_AP.glob(f"*/*{pos}*/CropSize_{crop_size}"), key=sort_fish_dsname)
    pbar = tqdm(total=len(img_paths), desc=f"[ Crop Images ] save_dir_Mix_AP ---copy---> save_dir_{pos}_only: ")

    for img_path in img_paths:
        
        palmskin_dsname = str(img_path).split(os.sep)[-2]
        pbar.desc = f"[ Crop Images ] save_dir_Mix_AP ---copy---> save_dir_{pos}_only ( {palmskin_dsname} ) : "
        pbar.refresh()
        
        original_path = img_path
        new_path = Path(str(original_path).replace("fish_dataset_horiz_cut_1l2_Mix_AP",
                                                   f"fish_dataset_horiz_cut_1l2_{pos}_only"))
        shutil.copytree(original_path, new_path)
        
        pbar.update(1)
        pbar.refresh()

    pbar.close()