import os
import sys
from glob import glob
from typing import List
from pathlib import Path
from datetime import datetime
import argparse

import json
import pandas as pd

import numpy as np

import cv2


# *** show images methods ***
# cv2.imshow("fish", fish)
# cv2.waitKey(0)
# plt_show.img_rgb("fish", fish)
# plt_show.img_gray("fish", fish)
# plt_show.img_by_channel("fish", fish)



def create_new_dir(path:str, end="\n"):
    if not os.path.exists(path):
        # if the demo_folder directory is not exist then create it.
        os.makedirs(path)
        print(f"path: '{path}' is created!{end}")



def gen_crop_img(img, crop_size:int, shift_region:float):
    
    """ 
    generate the crop images using 'crop_size' and 'shift_region'
    
    Args:
        img : the image you want to generate its crop images
        crop_size (int): output size/shape of an crop image
        shift_region (float): if 'shift_region' is passed, calculate the shift region while creating each cropped image, e.g. shift_region=1/3, the overlap region of each cropped image is 2/3
    """
    
    img_size = img.shape
    
    
    # *** calculate relation between img & crop_size ***
    
    # if 'shift_region' is passed, calculate the shift region while creating each cropped image, e.g. shift_region=1/3, the overlap region of each cropped image is 2/3
    DIV_PIECES = int(1/shift_region) # DIV_PIECES, abbr. of 'divide pieces'
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



def drop_too_dark(crop_img_list:List, intensity:int, drop_ratio:float):
    
    """ 
    drop the image which too many dark pixels
    
    Args:
        crop_img_list (List): a list contain 'gray images'
        crop_img_list_color (List): a list contain 'color(RGB) images'
        intensity (int): a threshold (image is grayscale) to define too dark or not 
        drop_ratio (float): a threshold (pixel_too_dark / all_pixel) to decide the crop image keep or drop, if drop_ratio < 0.5, keeps the crop image.
    """
    
    
    # check if the height and width of passed two lists with the same size
    img_size = crop_img_list[0].shape
    
    select_crop_img_list = []

    
    for i in range(len(crop_img_list)):
        # change to grayscale
        in_grayscale = cv2.cvtColor(crop_img_list[i], cv2.COLOR_BGR2GRAY)
        # count pixels too dark
        pixel_too_dark = np.sum( in_grayscale < intensity )
        dark_ratio = pixel_too_dark/(crop_size*crop_size)
        if dark_ratio < drop_ratio: # 有表皮資訊的
            select_crop_img_list.append(crop_img_list[i])


    return select_crop_img_list
    


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
    fish_names = []
    cropped_name_list = []
    #
    logs = []
    


    # *** Load Excel sheet as DataFrame(pandas) ***
    df_QcImg = pd.read_excel(xlsx_file, engine = 'openpyxl')
    # print(df_QcImg)
    df_A_names_list = df_QcImg["Experiment series name anterior (SP8)"].tolist()
    # print(type(df_A_names_list), len(df_A_names_list), df_A_names_list)
    df_P_names_list = df_QcImg["Experiment series name posterior (SP8)"].tolist()
    # print(type(df_P_names_list), len(df_P_names_list), df_P_names_list)
    assert len(df_A_names_list) == len(df_P_names_list), "number of anterior NOT match the posterior"
    #
    fish_names.extend(df_A_names_list)
    fish_names.extend(df_P_names_list)



    # *** Start process ***
    for fish_name in fish_names:
        
        
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
        # for path, e.g. ".\stacked_palmskin_tiff\20220727 CE012 palmskin_9dpf - Series002 fish 111 palmskin_9df_P_RGB only.tif"
        fish_name_list = fish_path.split(" ")
        print(fish_name_list)
        assert len(fish_name_list) == 9, "IMAGE_NAME Format Error!"
        #
        fish_id = f"{fish_name_list[5]}_{fish_name_list[6]}" # Extracting fish_id, e.g. "fish_1"
        #
        position_tag = fish_name_list[-2].split("_")[2] # Extracting position_tag, Anterior --> A, Posterior --> P
        # print(fish_id, position_tag)
        #
        save_dir_test = f"{save_dir}\\test"
        create_new_dir(save_dir_test)
        #
        # write cropped images
        for i in range(len(select_crop_img_list)):
            write_name = f"{fish_id}_{position_tag}_crop_{i}.tiff"
            write_path = os.path.join(save_dir_test, write_name)
            cropped_name_list.append(write_name)
            
            cv2.imwrite(write_path, select_crop_img_list[i])
            # cv2.imshow(write_name, select_crop_img_list[i])
            # cv2.waitKey(0)



        # *** Update log ***
        current_log = {
            "fish_id": fish_id,
            "number_of_crop": len(crop_img_list),
            "number_of_drop": len(crop_img_list) - len(select_crop_img_list),
            "number_of_saved": len(select_crop_img_list),
        }
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
    log_path_abs = f"{write_log_dir}/[log]{time_stamp}_using_(crop_img).json"
    df.to_json(log_path_abs, orient ='index', indent=2)
    print("\n", f"log save @ \n-> {log_path_abs}\n")
    
    
    
    # *** create DataFrame with cropped images name ***
    df_new_crop = pd.DataFrame()
    df_new_crop["image_name"] = cropped_name_list
    # save
    xlsx_save_path = f"{save_dir}/{save_dir}.xlsx"
    df_new_crop.to_excel(xlsx_save_path, index=False, engine = 'openpyxl')
    
    
    print("="*100, "\n", "process all complete !", "\n")