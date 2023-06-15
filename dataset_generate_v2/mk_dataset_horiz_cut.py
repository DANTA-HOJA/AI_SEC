import os
import sys
import shutil
import re
from colorama import Fore, Back, Style
from pathlib import Path
import toml

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import cv2

abs_module_path = Path("./../modules/").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from fileop import create_new_dir
from data.utils import get_fish_id_pos
from data.ProcessedDataInstance import ProcessedDataInstance
from dataset.utils import sort_fish_dsname

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
relative_path_in_fish_dir = processed_data_instance.check_palmskin_images_condition(palmskin_result_alias)

# -----------------------------------------------------------------------------------
# Generate `path_vars`

# dataset_dir
instance_name = processed_data_instance.instance_name
dataset_root = processed_data_instance.db_root.joinpath(processed_data_instance.dbpp_config["dataset_cropped_v2"])

save_dir_A_only = dataset_root.joinpath(instance_name, palmskin_result_alias, "fish_dataset_horiz_cut_1l2_A_only")
save_dir_P_only = dataset_root.joinpath(instance_name, palmskin_result_alias, "fish_dataset_horiz_cut_1l2_P_only")
save_dir_Mix_AP = dataset_root.joinpath(instance_name, palmskin_result_alias, "fish_dataset_horiz_cut_1l2_Mix_AP")

# check directory existence
exists_cnt = 0
for dir in [save_dir_A_only, save_dir_P_only, save_dir_Mix_AP]:
    if os.path.exists(dir): 
        print(f"{Fore.RED}{Back.BLACK} dir: '{dir}' already exists {Style.RESET_ALL}")
        exists_cnt += 1
assert exists_cnt == 0, f"{Fore.RED} Diretory already exists, process has been halted. {Style.RESET_ALL}\n"
create_new_dir(save_dir_A_only)
create_new_dir(save_dir_P_only)
create_new_dir(save_dir_Mix_AP)

# -----------------------------------------------------------------------------------
# Load `data.xlsx` as DataFrame(pandas) 

# `data.xlsx` existence has been checked by `processed_data_instance.check_palmskin_images_condition()`
df_xlsx :pd.DataFrame = pd.read_excel(processed_data_instance.data_xlsx_path, engine = 'openpyxl')

# -----------------------------------------------------------------------------------
# Do `Horizontal Cut`
    
# variables
save_dir_train = save_dir_Mix_AP.joinpath("train")
save_dir_test = save_dir_Mix_AP.joinpath("test")
rand_choice_result = {"up : train, down: test": 0, 
                      "up : test, down: train": 0}
palmskin_dnames = sorted(pd.concat([df_xlsx["Anterior (SP8, .tif)"], df_xlsx["Posterior (SP8, .tif)"]]), key=get_fish_id_pos)

pbar = tqdm(total=len(palmskin_dnames), desc=f"[ Horizontal_Cut ] : ")

for i, palmskin_dname in enumerate(palmskin_dnames):
    
    path = processed_data_instance.palmskin_preprocess_dir.joinpath(palmskin_dname, relative_path_in_fish_dir)
    img = cv2.imread(str(path))
    assert img.shape[0] == img.shape[1], "please pad the image to make it a square image."
    
    # horizontal cut (image -> up, down)
    half_position = int(img.shape[0]/2)
    img_up = img[0:half_position, :, :]
    img_down = img[half_position:half_position*2, :, :]
    
    # get `palmskin_dsname``
    fish_id, fish_pos = get_fish_id_pos(palmskin_dname)
    palmskin_dsname = f"fish_{fish_id}_{fish_pos}"
    pbar.desc = f"[ Horizontal_Cut ] {palmskin_dsname} : "
    pbar.refresh()
    
    if i%2 == 0: action = random_state.choice([True, False], size=1, replace=False)[0]
    else: action = not action
    
    if action:
        
        # if i%2 == 0: tqdm.write("")
        # tqdm.write(f"palmskin_dsname: '{palmskin_dsname}' --> up : test, down: train")
        
        # img_up -> test
        save_name = f"{palmskin_dsname}_U"
        dir = save_dir_test.joinpath(save_name)
        create_new_dir(dir, display_in_CLI=False)
        cv2.imwrite(str(dir.joinpath(f"{save_name}.tiff")), img_up)
        
        # img_down -> train
        save_name = f"{palmskin_dsname}_D"
        dir = save_dir_train.joinpath(save_name)
        create_new_dir(dir, display_in_CLI=False)
        cv2.imwrite(str(dir.joinpath(f"{save_name}.tiff")), img_down)
        
        rand_choice_result["up : test, down: train"] += 1

    else:
        
        # if i%2 == 0: tqdm.write("")
        # tqdm.write(f"palmskin_dsname: '{palmskin_dsname}' --> up : train, down: test")
        
        # img_up -> train
        save_name = f"{palmskin_dsname}_U"
        dir = save_dir_train.joinpath(save_name)
        create_new_dir(dir, display_in_CLI=False)
        cv2.imwrite(str(dir.joinpath(f"{save_name}.tiff")), img_up)
        
        # img_down -> test
        save_name = f"{palmskin_dsname}_D"
        dir = save_dir_test.joinpath(save_name)
        create_new_dir(dir, display_in_CLI=False)
        cv2.imwrite(str(dir.joinpath(f"{save_name}.tiff")), img_down)
        
        rand_choice_result["up : train, down: test"] += 1
    
    pbar.update(1)
    pbar.refresh()

pbar.close()
print(f"rand_choice_result = {rand_choice_result}\n")

# -----------------------------------------------------------------------------------
# save_dir_Mix_AP ---copy---> save_dir_{pos}_only

for pos in ["A", "P"]:
    
    img_paths = sorted(save_dir_Mix_AP.glob(f"*/*{pos}*"), key=sort_fish_dsname)
    pbar = tqdm(total=len(img_paths), desc=f"[ Horizontal_Cut ] save_dir_Mix_AP ---copy---> save_dir_{pos}_only: ")

    for img_path in img_paths:
        
        palmskin_dsname = str(img_path).split(os.sep)[-1]
        pbar.desc = f"[ Horizontal_Cut ] save_dir_Mix_AP ---copy---> save_dir_{pos}_only ( {palmskin_dsname} ) : "
        pbar.refresh()
        
        original_path = img_path
        new_path = Path(str(original_path).replace("fish_dataset_horiz_cut_1l2_Mix_AP",
                                                   f"fish_dataset_horiz_cut_1l2_{pos}_only"))
        shutil.copytree(original_path, new_path)
        
        pbar.update(1)
        pbar.refresh()

    pbar.close()