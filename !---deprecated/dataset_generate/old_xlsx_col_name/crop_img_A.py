import os
import sys
import argparse
from glob import glob
from typing import List
from pathlib import Path
from datetime import datetime
from collections import Counter
import json

import cv2
import pandas as pd
import numpy as np

sys.path.append("/home/rime97410000/ZebraFish_Code/ZebraFish_AP_POS/modules") # add path to scan customized module
from fileop import create_new_dir
from dataset.utils import gen_crop_img, drop_too_dark



def get_args():
    
    parser = argparse.ArgumentParser(description="zebrafish project: crop images into small pieces")
    parser.add_argument(
        "--xlsx_file",
        type=str,
        required=True,
        help="The path of the Excel sheet.",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="The path of the original images.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=224,
        help="the size to crop images.",
    )
    parser.add_argument(
        "--crop_shift_region",
        type=float,
        default=1/2,
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
        "--save_dir",
        type=str,
        required=True,
        help="the path to save the cropped images.",
    )
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = get_args()
    
    
    # *** Variable ***
    # args variable
    xlsx_file = args.xlsx_file
    source_dir = args.source_dir # input by user, required=True.
    crop_size = args.crop_size # size of the crop image.
    crop_shift_region = args.crop_shift_region # if 'shift_region' is passed, calculate the shift region while creating each cropped image,
    #                                            e.g. shift_region=1/3, the overlapping region of each cropped image is 2/3.                  
    intensity = args.intensity # a threshold to define too dark or not.
    drop_ratio = args.drop_ratio # a threshold (pixel_too_dark / all_pixel) to decide the crop image keep or drop, 
    #                              if drop_ratio < 0.5, keeps the crop image.
    save_dir = args.save_dir # input by user, required=True.
    print("")
    create_new_dir(save_dir)
    #
    # variable
    logs = []
    


    # *** Load Excel sheet as DataFrame(pandas) ***
    df_input_xlsx = pd.read_excel(xlsx_file, engine = 'openpyxl', sheet_name="class_by_length")
    # print(df_input_xlsx)
    #
    df_A_names_list = df_input_xlsx["Experiment series name anterior (SP8)"].tolist()
    df_class_list = df_input_xlsx["class"].tolist()
    #
    assert len(df_A_names_list) == len(df_class_list), "length of 'A_name_list' and 'class_list' misMatch"
    print(f"len(df_A_names_list) = len(df_class_list) = {len(df_class_list)}\n")



    # *** Start process ***
    for i, fish_name in enumerate(df_A_names_list):
        
        
        # *** Print CMD section divider ***
        print("="*100, "\n")
        
        
        # *** Load image ***
        fish_path = f"{source_dir}/{fish_name}.tif"
        fish = cv2.imread(fish_path)
        
        
        # *** If the image is horizontal, rotate the image ***
        if fish.shape[0] < fish.shape[1]: # 照片不是直的，要轉向
            fish = cv2.rotate(fish, cv2.ROTATE_90_CLOCKWISE)
        
        
        # *** Check orient by user ***
        fish_resize_to_display = cv2.resize(fish, (int(fish.shape[1]/5), int(fish.shape[0]/5)))
        # cv2.imshow("fish_resize_to_display", fish_resize_to_display)
        # cv2.waitKey(0)
        
        
        # *** Test cv2.split ***
        b, g, r = cv2.split(fish)
        assert (fish[:,:,0] == b).all(), "Blue channel NOT match!"
        assert (fish[:,:,1] == g).all(), "Green channel NOT match!"
        assert (fish[:,:,2] == r).all(), "Red channel NOT match!"



        # *** Crop original image to small images ***
        crop_img_list = gen_crop_img(fish, crop_size, crop_shift_region)


        # *** Drop image which too many dark pixels ***
        select_crop_img_list = drop_too_dark(crop_img_list, intensity, drop_ratio)



        # *** Save select_crop_img ***
        ## for path, e.g. ".\stacked_palmskin_tiff\20220727 CE012 palmskin_9dpf - Series002 fish 111 palmskin_9df_P_RGB only.tif"
        fish_name_list = fish_name.split(" ")
        print(fish_name_list)
        assert len(fish_name_list) == 9, "IMAGE_NAME Format Error!"
        #
        ## Extracting fish_id, e.g. "fish_1"
        fish_id = f"{fish_name_list[5]}_{fish_name_list[6]}"
        #
        ## Extracting position_tag, Anterior --> A, Posterior --> P
        position_tag = fish_name_list[-2].split("_")[2] 
        #
        ## Looking up the class of current fish
        fish_size = df_class_list[i]
        #
        # print(fish_size, fish_id, position_tag)
        #
        save_dir_size = f"{save_dir}/{fish_size}"
        create_new_dir(save_dir_size)
        #
        ## write cropped images
        for j in range(len(select_crop_img_list)):
            write_name = f"{fish_size}_{fish_id}_{position_tag}_crop_{j}.tiff"
            write_path = os.path.join(save_dir_size, write_name)
            
            cv2.imwrite(write_path, select_crop_img_list[j])
            # cv2.imshow(write_name, select_crop_img_list[j])
            # cv2.waitKey(0)



        # *** Update log ***
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
        print(all_class)
        #
        ## creat fish_size column in current_log
        current_log["Class Count"] = ""
        for size in all_class: current_log[size] = 0
        #
        ## Update # of saved images to current_log 
        current_log[fish_size] = len(select_crop_img_list)
        #
        logs.append(current_log)
        # print current log in command
        print(json.dumps(current_log, indent=2), "\n")
    
    
    
    # *** Change logs into Dataframe and show in command ***
    df = pd.DataFrame(logs)
    df.loc['TOTAL'] = df.select_dtypes(np.number).sum()
    print("="*100, "\n")
    print(df, "\n")
    
    
    
    # *** Save logs ***
    write_log_dir = save_dir
    create_new_dir(write_log_dir)
    #
    # get time to as file name
    time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S') # get time to as file name
    log_path_abs = f"{write_log_dir}/[log]{time_stamp}_using_(crop_img_A).json"
    df.to_json(log_path_abs, orient ='index', indent=2)
    print("\n", f"log save @ \n-> {log_path_abs}\n")
    
    
    print("="*100, "\n", "process all complete !", "\n")