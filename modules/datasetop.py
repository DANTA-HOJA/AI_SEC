import os
import sys
import re
import argparse
from typing import List, Dict, Tuple
from glob import glob
import json
import yaml

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import cv2

sys.path.append(r"C:\Users\confocal_microscope\Desktop\ZebraFish_AP_POS\modules") # add path to scan customized module
from fileop import create_new_dir



def get_args():
    
    parser = argparse.ArgumentParser(description="zebrafish project: crop images into small pieces")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="The config file in 'yaml' format for 'mk_dataset_horiz_cut.py'.",
    )
    
    return parser.parse_args()



def gen_dataset_param_name(xlsx_file:str, crop_size:int, shift_region:str, intensity:int, drop_ratio:float, random_seed:int=None) -> str:
    """To generate dataset's name corresponing to the passing parameters.
    
    Args:
        xlsx_file (str):                e.g. "{4CLS_BY_SurfStDev}_data.xlsx" ---> SURF4C
        crop_size (int):                e.g.       512                       ---> CRPS512
        shift_region (str):             e.g.      "1/4"                      ---> SF14
        intensity (int):                e.g.        20                       ---> INT20
        drop_ratio (float):             e.g.        0.3                      ---> DRP30
        random_seed (int, optional):    e.g.       2022                      ---> RS2022. Defaults to None.

    Raises:
        ValueError: "Numerator of 'shift_region' needs to be 1"

    Returns:
        str:
        e.g. 'DS_SURF4C_CRPS512_SF14_INT20_DRP30' or 'DS_SURF4C_CRPS512_SF14_INT20_DRP30_RS2022'
    
    """
    
    # Converting... "xlsx_file" 
    xlsx_file_split = re.split("{|_|}", xlsx_file)
    n_class = xlsx_file_split[1].replace("CLS", "C")
    used_feature = xlsx_file_split[3].replace("StDev", "").upper()
    
    # Converting... "shift_region"
    fraction = shift_region.split("/")
    if int(fraction[0]) == 1: fraction = f"{fraction[0]}{fraction[1]}"
    else: raise ValueError("Numerator of 'shift_region' needs to be 1")
    
    # Converting... "drop_ratio"
    ratio = int(drop_ratio*100)
     
    gen_name = f"DS_{used_feature}{n_class}_CRPS{str(crop_size)}_SF{fraction}_INT{str(intensity)}_DRP{str(ratio)}"
    
    if random_seed is None: return gen_name
    else: return f"{gen_name}_RS{str(random_seed)}"



def gen_crop_img(img:cv2.Mat, crop_size:int, shift_region:str="1/1") -> List[cv2.Mat]:
    """generate the crop images using 'crop_size' and 'shift_region'

    Args:
        img (cv2.Mat): the image you want to generate its crop images
        crop_size (int): output size/shape of an crop image
        shift_region (str): if 'shift_region' is passed, calculate the offset distance between cropped images, e.g. shift_region=1/3, the overlap region of each cropped image is 2/3

    Returns:
        List[cv2.Mat]
    """
    
    img_size = img.shape
    
    
    # *** calculate relation between img & crop_size ***
    
    fraction = shift_region.split("/")
    if int(fraction[0]) == 1: DIV_PIECES = int(fraction[1]) # DIV_PIECES, abbr. of 'divide pieces'
    else: raise ValueError("Numerator of 'shift_region' needs to be 1")
    # print(DIV_PIECES, "\n")
    # input()
    
    # Height
    quotient_h = int(img_size[0]/crop_size) # Height Quotient
    remainder_h = img_size[0]%crop_size # Height Remainder
    h_st_idx = remainder_h # image 只有上方有黑色
    h_crop_idxs = [(h_st_idx + int((crop_size/DIV_PIECES)*i)) 
                           for i in range(quotient_h*DIV_PIECES+1)]
    # print(f"h_crop_idxs = {h_crop_idxs}", f"len = {len(h_crop_idxs)}", "\n")
    # input()
    
    # Width
    quotient_w = int(img_size[1]/crop_size) # Width Quotient
    remainder_w = img_size[1]%crop_size # Width Remainder
    w_st_idx = int(remainder_w/2) # image 左右都有黑色，平分
    w_crop_idxs = [(w_st_idx + int((crop_size/DIV_PIECES)*i))
                           for i in range(quotient_w*DIV_PIECES+1)]
    # print(f"w_crop_idxs = {w_crop_idxs}", f"len = {len(w_crop_idxs)}", "\n")
    # input()
    
    
    # *** crop original image to small images ***
    
    # do crop 
    crop_img_list = []
    for i in range(len(h_crop_idxs)-DIV_PIECES):
        for j in range(len(w_crop_idxs)-DIV_PIECES):
            # print(h_crop_idxs[i], h_crop_idxs[i+DIV_PIECES], "\n", w_crop_idxs[j], w_crop_idxs[j+DIV_PIECES], "\n")
            crop_img_list.append(img[h_crop_idxs[i]:h_crop_idxs[i+DIV_PIECES], w_crop_idxs[j]:w_crop_idxs[j+DIV_PIECES], :])
    
    
    return crop_img_list



def drop_too_dark(crop_img_list:List[cv2.Mat], intensity:int, drop_ratio:float) -> Tuple[List[cv2.Mat], List[cv2.Mat]] : 
    """drop the image which too many dark pixels

    Args:
        crop_img_list (List[cv2.Mat]): a list contain 'gray images'
        intensity (int): a threshold (image is grayscale) to define too dark or not 
        drop_ratio (float): a threshold (pixel_too_dark / all_pixel) to decide the crop image keep or drop, if drop_ratio < 0.5, keeps the crop image.

    Returns:
        Tuple[List[cv2.Mat], List[cv2.Mat]]: first List is 'select_crop_img_list' , second List is 'drop_crop_img_list'
    """
    
    select_crop_img_list = []
    drop_crop_img_list = []

    for i in range(len(crop_img_list)):
        
        # change to HSV/HSB and get V_channel (Brightness)
        brightness = cv2.cvtColor(crop_img_list[i], cv2.COLOR_BGR2HSV_FULL)[:,:,2]
        # count pixels too dark
        pixel_too_dark = np.sum( brightness <= intensity )
        # get image size
        img_size = crop_img_list[i].shape
        # calculate ratio
        dark_ratio = pixel_too_dark/(img_size[0]*img_size[1])
        if dark_ratio >= drop_ratio: # 減少 model 學習沒有表皮資訊的位置的可能性
            drop_crop_img_list.append((i, crop_img_list[i], dark_ratio))
        else: 
            select_crop_img_list.append((i, crop_img_list[i], dark_ratio))

    return select_crop_img_list, drop_crop_img_list



def save_crop_img(crop_img_list:List[Tuple[int, cv2.Mat]], darkratio_log:Dict[str, float],
                  save_dir:str, crop_img_desc:str, fish_size:str, fish_id:str, fish_pos:str,
                  tqdm_process:tqdm, tqdm_overwrite_desc:str=None):
    
    
    # adjust settings of 'tqdm_process'
    if tqdm_overwrite_desc is not None: tqdm_process.desc  = tqdm_overwrite_desc
    tqdm_process.n     = 0   # current value
    tqdm_process.total = len(crop_img_list)
    tqdm_process.refresh()
    
    # write crop images
    for item in crop_img_list: # item : ( crop_idx , crop_img, dark_ratio )
        
        save_dir_size = os.path.join(save_dir, crop_img_desc, fish_size)
        create_new_dir(save_dir_size, display_in_CLI=False)
        write_name = f"{fish_size}_fish_{fish_id}_{fish_pos}_{crop_img_desc}_{item[0]}"
        write_path = os.path.join(save_dir_size, f"{write_name}.tiff")

        crop_img = item[1] # convenience to debug preview
        cv2.imwrite(write_path, crop_img)
        # cv2.imshow(write_name, select_crop_img)
        # cv2.waitKey(0)
        
        if write_name not in darkratio_log: darkratio_log[write_name] = item[2]
        
        tqdm_process.update(1)
        tqdm_process.refresh()



def append_log(logs:List[Dict], fish_size:str, fish_id:str, fish_pos:str, selected_part:str, all_class:List[str],
               crop_img_list:List[cv2.Mat], select_crop_img_list:List[cv2.Mat], drop_crop_img_list:List[cv2.Mat]):
    
    
    current_log = {
        "fish_name (dataset)": f"{fish_size}_fish_{fish_id}_{fish_pos}",
        "selected part": selected_part,
        #
        "number_of_crop": len(crop_img_list),
        "number_of_drop": len(drop_crop_img_list),
        "number_of_saved": len(select_crop_img_list),
    }
    #
    ## creat fish_size columns in current_log
    current_log["Class Count"] = ""
    for size in all_class: current_log[size] = 0
    #
    ## update # of saved images to current_log
    current_log[fish_size] = len(select_crop_img_list)
    #
    logs.append(current_log)
    # print current log in command
    # print(json.dumps(current_log, indent=2), "\n")



def save_dataset_logs(logs:List[Dict], save_dir:str, log_desc:str, script_name:str, time_stamp:str, CLI_desc:str="", show_df:bool=True):
    
    df = pd.DataFrame(logs)
    df.loc['TOTAL'] = df.select_dtypes(np.number).sum()
    # print("="*100, "\n")
    if show_df: print(f"\n{CLI_desc}\n", df, "\n")
    
    # *** Save logs ***
    log_path_abs = os.path.normpath(f"{save_dir}/{{{log_desc}}}_{{{script_name}}}_{time_stamp}.xlsx")
    df.to_excel(log_path_abs, engine="openpyxl")
    print("\n", f"log save @ \n-> {log_path_abs}\n")



def gen_train_selected_summary(dir_path:str, all_class:List[str]):
    
    class_count = {}
    
    for cls in all_class:
        selected_images_list = glob(os.path.normpath(f"{dir_path}/train/selected/{cls}/*.tiff"))
        class_count[cls] = len(selected_images_list)
    
    with open(os.path.normpath(f"{dir_path}/{{Logs}}_train_selected_summary.log"), mode="w") as f_writer:
        f_writer.write(json.dumps(class_count, indent=4))



def save_dataset_config(save_path:str, config:Dict):
    
    with open(os.path.normpath(f"{save_path}/dataset_config.yaml"), mode="w") as f_writer:
        f_writer.write(yaml.dump(config))



def sortFishNameForDataset(fish_name_string:str) -> Tuple[int, str, int]:
    
    if os.sep in fish_name_string: fish_name_for_dataset = fish_name_string.split(os.sep)[-1].split(".")[0]
    else: fish_name_for_dataset = fish_name_string.split(".")[0]
    
    name_split_list = re.split(" |_|-", fish_name_for_dataset) # example_list : ['L', 'fish', '111', 'A', 'selected', '0']
    
    return name_split_list[0], int(name_split_list[2]), name_split_list[3], int(name_split_list[5])



def save_dark_ratio_log(log:Dict[str, float], save_dir:str, log_desc):
    
    with open(os.path.normpath(f"{save_dir}/{{{log_desc}}}_dark_ratio.log"), mode="w") as f_writer: 
        f_writer.write(json.dumps(log, indent=4))