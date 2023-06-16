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
from dataset.utils import gen_dataset_param_name, sort_fish_dsname, drop_too_dark, \
                          xlsx_file_name_parser
from misc.utils import get_target_str_idx_in_list
from misc.CLIDivider import CLIDivider
cli_divider = CLIDivider()
cli_divider.process_start(use_tqdm=True)

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
dyn_train    = config["gen_param"]["dynamic_training"]
random_state = np.random.RandomState(seed=random_seed)

# -----------------------------------------------------------------------------------
# Initialize a `ProcessedDataInstance` object

processed_data_instance = ProcessedDataInstance(config_dir, processed_inst_desc)
# check images are existing and readable
# relative_path_in_fish_dir = processed_data_instance.check_palmskin_images_condition(palmskin_result_alias)

# -----------------------------------------------------------------------------------
# Generate `path_vars`

# clustered_xlsx_path
clustered_desc = re.split("{|}", clustered_xlsx_file)[1]
clustered_xlsx_path = processed_data_instance.clustered_xlsx_paths_dict[clustered_desc]
print(f"clustered_xlsx ( load from ) : '{clustered_xlsx_path}'\n")

# dataset_dir
instance_name = processed_data_instance.instance_name
dataset_root = processed_data_instance.db_root.joinpath(processed_data_instance.dbpp_config["dataset_cropped_v2"])

save_dir_Mix_AP = dataset_root.joinpath(instance_name, palmskin_result_alias, "fish_dataset_horiz_cut_1l2_Mix_AP")
assert save_dir_Mix_AP.exists(), f"{Fore.RED}{Back.BLACK} Cant' find diretory: '{save_dir_Mix_AP}' {Style.RESET_ALL}\n"

# crop_dir_name
param_name_dict:dict = gen_dataset_param_name(clustered_xlsx_file, crop_size, shift_region, intensity, drop_ratio, 
                                              random_seed, dict_format=True)
crop_dir_name =  f"{param_name_dict['crop_size']}_{param_name_dict['shift_region']}"

# dataset_xlsx_dir
classif_strategy = xlsx_file_name_parser(clustered_xlsx_file)
dataset_xlsx_dir = save_dir_Mix_AP.joinpath(classif_strategy)
create_new_dir(dataset_xlsx_dir)

# dataset_xlsx_path
param_name_str:str = gen_dataset_param_name(clustered_xlsx_file, crop_size, shift_region, intensity, drop_ratio, 
                                            random_seed, dict_format=False)
if dyn_train: param_name_str = param_name_str.replace(f"{param_name_dict['intensity']}_{param_name_dict['drop_ratio']}", "DYNTRAIN")
dataset_xlsx_path = dataset_xlsx_dir.joinpath(f"{param_name_str}.xlsx")
print(f"dataset_xlsx ( plan to save @ ) : '{dataset_xlsx_path}'\n")
assert not dataset_xlsx_path.exists(), f"{Fore.RED}{Back.BLACK} `dataset_xlsx` already exists: '{dataset_xlsx_path}' {Style.RESET_ALL}\n"

# -----------------------------------------------------------------------------------
# Load `clustered_xlsx` as DataFrame(pandas)

df_clustered_xlsx: pd.DataFrame = pd.read_excel(clustered_xlsx_path, engine = 'openpyxl')

# -----------------------------------------------------------------------------------
# Do `Create Dataset XLSX`

# add column: `fish_id`
df_clustered_xlsx['fish_id'] = df_clustered_xlsx['Anterior (SP8, .tif)'].apply(lambda x: get_fish_id_pos(x)[0])

if dyn_train: 
    img_paths = []
    img_paths.extend(sorted(save_dir_Mix_AP.glob(f"train/*/*.tiff"), key=sort_fish_dsname))
    img_paths.extend(sorted(save_dir_Mix_AP.glob(f"test/*/{crop_dir_name}/*.tiff"), key=sort_fish_dsname))
else: img_paths = sorted(save_dir_Mix_AP.glob(f"**/{crop_dir_name}/*.tiff"), key=sort_fish_dsname)
df_dataset = None

pbar = tqdm(total=len(img_paths), desc=f"[ Create Dataset XLSX ] : ")

for img_path in img_paths:
    
    img_path_str = str(img_path)
    img_path_split = img_path_str.split(os.sep)
    
    # image_name
    image_name = img_path_split[-1].split(".")[0]
    pbar.desc = f"[ Create Dataset XLSX ] {image_name} : "
    pbar.refresh()
    
    # dsname ( without `crop_idx` )
    target_idx = get_target_str_idx_in_list(img_path_split, "fish_dataset_horiz_cut_1l2_Mix_AP")
    parent_dsname = img_path_split[target_idx+2]
    parent_dsname_split = parent_dsname.split("_")
    fish_id     = int(parent_dsname_split[1])
    fish_pos    = parent_dsname_split[2]
    cut_section = parent_dsname_split[3]
    
    # class
    df_filtered_rows = df_clustered_xlsx[df_clustered_xlsx['fish_id'] == fish_id]
    df_filtered_rows = df_filtered_rows.reset_index(drop=True)
    fish_class = df_filtered_rows.loc[0, "class"]
    
    # preserve / discard ( if dyn_train == False )
    if dyn_train:
        state = "---"
        darkratio = "---"
    else:
        img = cv2.imread(img_path_str)
        select, drop = drop_too_dark([img], intensity, drop_ratio)
        if select is None: assert drop, "Either `select` or `drop` needs to be empty"
        if drop is None: assert select, "Either `select` or `drop` needs to be empty"
        
        if select:
            state = "preserve"
            darkratio = select[0][2]
            
        if drop:
            state = "discard"
            darkratio = drop[0][2]
    
    
    # create `temp_dict`
    temp_dict = {}
    # -------------------------------------------------------
    temp_dict["image_name"] = image_name
    temp_dict["class"] = fish_class
    # -------------------------------------------------------
    temp_dict["parent (dsname)"] = parent_dsname
    temp_dict["fish_id"]         = fish_id
    temp_dict["fish_pos"]        = fish_pos
    temp_dict["cut_section"]     = cut_section
    # -------------------------------------------------------
    temp_dict["dataset"]   = img_path_split[target_idx+1]
    temp_dict["darkratio"] = darkratio
    temp_dict["state"]     = state
    # -------------------------------------------------------
    temp_dict["path"] = img_path.relative_to(processed_data_instance.db_root)
    # -------------------------------------------------------
    temp_df = pd.DataFrame(temp_dict, index=[0])
    
    # add to `df_dataset`
    if df_dataset is None: df_dataset = temp_df
    else: df_dataset = pd.concat([df_dataset, temp_df], ignore_index=True)
    
    pbar.update(1)
    pbar.refresh()

pbar.close()

# -----------------------------------------------------------------------------------
# Save `dataset_xlsx`

print("\nSaving `dataset_xlsx`... ")
df_dataset.to_excel(dataset_xlsx_path)
print(f"{Fore.GREEN}{Back.BLACK} Done! {Style.RESET_ALL}")

# -----------------------------------------------------------------------------------
cli_divider.process_completed()