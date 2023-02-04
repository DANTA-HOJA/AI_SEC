import os
import sys
from glob import glob
from typing import List
from pathlib import Path
from datetime import datetime
import argparse
from tqdm.auto import tqdm

import json
import pandas as pd

import numpy as np
from collections import Counter

import cv2

from dataset_generate_functions import create_new_dir, gen_dataset_name, gen_crop_img, drop_too_dark


# *** show images methods ***
# cv2.imshow("fish", fish)
# cv2.waitKey(0)
# plt_show.img_rgb("fish", fish)
# plt_show.img_gray("fish", fish)
# plt_show.img_by_channel("fish", fish)



def get_args():
    
    parser = argparse.ArgumentParser(description="zebrafish project: crop images into small pieces")
    parser.add_argument(
        "--data_root_path",
        type=str,
        required=True,
        help="The root path of the data.",
    )
    parser.add_argument(
        "--xlsx_file",
        type=str,
        required=True,
        help=r"The name of the Excel book, under 'data_root_path/{Modify}_xlsx'",
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        required=True,
        help="The 'sheet_name' in 'xlsx_file' contain standard deviation classify result.",
    )
    parser.add_argument(
        "--stacked_palmskin_dir",
        type=str,
        required=True,
        help="The folder name of the 'stacked palmskin' images, , under 'data_root_path'",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=224,
        help="the size to crop images.",
    )
    parser.add_argument(
        "--crop_shift_region",
        type=str,
        default="1/4",
        help="the overlapping region between each cropped image.",
    )
    parser.add_argument(
        "--intensity",
        type=int,
        default=20,
        help="a threshold to define too dark or not.",
    )
    parser.add_argument(
        "--drop_ratio",
        type=float,
        default=0.5,
        help="a threshold (pixel_too_dark / all_pixel) to decide the crop image keep or drop.",
    )
    parser.add_argument(
        "--train_and_test",
        type=str,
        required=True,
        help="when assigned 'P/A', it indicates Posterior is TrainSet, Anterior is TestSet, and vice versa.",
    )
    parser.add_argument(
        "--dataset_root_path",
        type=str,
        required=True,
        help="the root path of datasets to save the 'cropped images'.",
    )
    
    args = parser.parse_args()
    
    # Check arguments
    fraction = args.crop_shift_region.split("/")
    if int(fraction[0]) != 1: raise ValueError("Numerator of 'crop_shift_region' needs to be 1")
    
    return args

    
    
if __name__ == "__main__":

    args = get_args()
    
    
    # *** Variable ***
    # args variable
    data_root_path = args.data_root_path
    data_name = data_root_path.split(os.sep)[-1]
    #
    xlsx_file = args.xlsx_file
    xlsx_file_fullpath = os.path.join(data_root_path, r"{Modify}_xlsx", xlsx_file)
    sheet_name = args.sheet_name
    #
    stacked_palmskin_dir = os.path.join(data_root_path, args.stacked_palmskin_dir)
    #
    crop_size = args.crop_size # size of the crop image.
    crop_shift_region = args.crop_shift_region # if 'shift_region' is passed, calculate the shift region while creating each cropped image,
                                               # e.g. shift_region=1/3, the overlapping region of each cropped image is 2/3.                  
    intensity = args.intensity # a threshold to define too dark or not.
    drop_ratio = args.drop_ratio # a threshold (pixel_too_dark / all_pixel) to decide the crop image keep or drop, 
                                 # if drop_ratio < 0.5, keeps the crop image.
    #
    train_and_test_no_slash = args.train_and_test.replace("/", "")
    train_and_test = args.train_and_test.split("/")
    train_and_test = {"train": train_and_test[0], "test":train_and_test[1]}
    #
    gen_name = gen_dataset_name(xlsx_file, crop_size, crop_shift_region, intensity, drop_ratio)
    save_dir = os.path.join(args.dataset_root_path, data_name, f"fish_dataset_simple_crop_{train_and_test_no_slash}", gen_name)
    print("")
    create_new_dir(save_dir)
    #
    # variable
    logs = []



    # *** Load Excel sheet as DataFrame(pandas) ***
    df_input_xlsx :pd.DataFrame = pd.read_excel(xlsx_file_fullpath, engine = 'openpyxl', sheet_name=sheet_name)
    # print(df_input_xlsx)
    
    
    for key, value in train_and_test.items():
        
        # *** Print CMD section divider ***
        print("="*100, "\n")
        print(f"{key} : {value}\n")


        if value == "A" : df_palmskin_list = df_input_xlsx["Anterior (SP8, .tif)" ].tolist()
        if value == "P" : df_palmskin_list = df_input_xlsx["Posterior (SP8, .tif)"].tolist()
        df_class_list = df_input_xlsx["class"].tolist()

        assert len(df_palmskin_list) == len(df_class_list), "length of 'palmskin_list' and 'class_list' misMatch"
        print(f"len(df_palmskin_list) = len(df_class_list) = {len(df_class_list)}\n")

        save_dir_set = os.path.join(save_dir, key)
        create_new_dir(save_dir_set)


        # *** Create progress bar ***
        pbar_n_fish = tqdm(total=len(df_palmskin_list), desc=f"Cropping {key}({value})")
        pbar_n_save = tqdm(desc="Saving... ")
        
        
        # *** Start process ***
        for i, fish_name in enumerate(df_palmskin_list):
            
            
            # *** Load image ***
            fish_path = f"{stacked_palmskin_dir}/{fish_name}.tif"
            fish = cv2.imread(fish_path)
            ## image processing
            fish = cv2.medianBlur(fish, 3)
            
            
            # *** If the image is horizontal, rotate the image ***
            if fish.shape[0] < fish.shape[1]: # 照片不是直的，要轉向
                fish = cv2.rotate(fish, cv2.ROTATE_90_CLOCKWISE)
            ## Check orient by user
            fish_resize_to_display = cv2.resize(fish, (int(fish.shape[1]/5), int(fish.shape[0]/5)))
            # cv2.imshow("fish_resize_to_display", fish_resize_to_display)
            # cv2.waitKey(0)
            
            
            # *** Test cv2.split *** ( just for fun )
            b, g, r = cv2.split(fish)
            assert (fish[:,:,0] == b).all(), "Blue channel NOT match!"
            assert (fish[:,:,1] == g).all(), "Green channel NOT match!"
            assert (fish[:,:,2] == r).all(), "Red channel NOT match!"


            # *** Crop original image to small images ***
            crop_img_list = gen_crop_img(fish, crop_size, crop_shift_region)


            # *** Drop image which too many dark pixels ***
            select_crop_img_list, _ = drop_too_dark(crop_img_list, intensity, drop_ratio)


            # *** Extracting / Looking up the information on current fish ***
            ## path, e.g. ".\stacked_palmskin_RGB\20220727_CE012_palmskin_9dpf - Series002_fish_111_P_RGB"
            fish_name = fish_name.replace(" ", "_")
            fish_name_list = fish_name.split("_")
            # print(fish_name_list)
            assert len(fish_name_list) == 10, "IMAGE_NAME Format Error!"
            #
            ## extracting fish_id, e.g. "fish_1"
            fish_id = f"{fish_name_list[6]}_{fish_name_list[7]}"
            #
            ## extracting position_tag, Anterior --> A, Posterior --> P
            position_tag = fish_name_list[-2]
            #
            ## looking up the class of current fish
            fish_size = df_class_list[i]
            #
            save_dir_size = f"{save_dir_set}/{fish_size}"
            create_new_dir(save_dir_size, use_tqdm=True)
            #
            # print(fish_size, fish_id, position_tag)
            
            
            # *** Save 'select_crop_img' ***
            ## adjust settings of 'pbar_n_save'
            pbar_n_save.n     = 0   # current value
            pbar_n_save.total = len(select_crop_img_list)
            pbar_n_save.desc  = f"Saving... '{fish_size}_{fish_id}_{position_tag}' "
            pbar_n_save.refresh()
            #
            ## write cropped images
            for j in range(len(select_crop_img_list)):
                write_name = f"{fish_size}_{fish_id}_{position_tag}_crop_{j}.tiff"
                write_path = os.path.join(save_dir_size, write_name)

                select_crop_img = select_crop_img_list[j] # convenience to debug preview
                cv2.imwrite(write_path, select_crop_img)
                # cv2.imshow(write_name, select_crop_img)
                # cv2.waitKey(0)
                
                pbar_n_save.update(1)
                pbar_n_save.refresh()


            # *** Update log of current fish ***
            current_log = {
                "fish_id": fish_id,
                #
                "number_of_crop": len(crop_img_list),
                "number_of_drop": len(crop_img_list) - len(select_crop_img_list),
                "number_of_saved": len(select_crop_img_list),
            }
            #
            ## create class_map for log count
            all_class = Counter(df_class_list)
            all_class = sorted(list(all_class.keys()))
            # print(all_class)
            #
            ## creat fish_size column in current_log
            current_log["Class Count"] = ""
            for size in all_class: current_log[size] = 0
            #
            ## update # of saved images to current_log
            current_log[fish_size] = len(select_crop_img_list)
            #
            logs.append(current_log)
            # print current log in command
            # print(json.dumps(current_log, indent=2), "\n")
            
            
            pbar_n_fish.update(1)
            pbar_n_fish.refresh()
        
        pbar_n_fish.close()
        pbar_n_save.close()
        
        
        # *** Change logs into Dataframe and show in command ***
        df = pd.DataFrame(logs)
        df.loc['TOTAL'] = df.select_dtypes(np.number).sum()
        # print("="*100, "\n")
        print("\n\n", df, "\n")
        
        
        # *** Save logs ***
        ## get time to as file name
        time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S') # get time to as file name
        log_path_abs = f"{save_dir}/{{log_{value}_{key}}}_{time_stamp}_using_(mk_dataset_simple_crop).json"
        df.to_json(log_path_abs, orient ='index', indent=2)
        print("\n", f"log save @ \n-> {log_path_abs}\n")


    print("="*100, "\n", "process all complete !", "\n")