import os
import sys
import argparse
from glob import glob
from typing import List
from pathlib import Path
from datetime import datetime
from collections import Counter
from copy import deepcopy
from math import floor
import traceback
import json

import cv2
import pandas as pd
import numpy as np

sys.path.append("/home/rime97410000/ZebraFish_Code/ZebraFish_AP_POS/modules") # add path to scan customized module
from fileop import create_new_dir
from datasetop import gen_crop_img, drop_too_dark



def get_args():
    
    parser = argparse.ArgumentParser(description="zebrafish project: crop images into small pieces")
    parser.add_argument(
        "--xlsx_file",
        type=str,
        required=True,
        help="The path of the Excel sheet.",
    )
    parser.add_argument(
        "--sheet_name",
        type=str,
        required=True,
        help="The sheet_name contain standard deviation classify result.",
    )
    parser.add_argument(
        "--select_column",
        type=str,
        required=True,
        help="anterior or posterior.",
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
        "--random_seed",
        type=int,
        default=2022,
        help="random seed, for choosing upper or lower as train part.",
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
    sheet_name = args.sheet_name
    select_column = args.select_column
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
    random_seed = args.random_seed
    np.random.seed(random_seed)
    #
    # variable
    trainset_logs = []
    testset_logs = []



    # *** Load Excel sheet as DataFrame(pandas) ***
    df_input_xlsx = pd.read_excel(xlsx_file, engine = 'openpyxl', sheet_name=sheet_name)
    # print(df_input_xlsx)
    #
    xlsx_names_list = df_input_xlsx[f"Experiment series name {select_column} (SP8)"].tolist()
    xlsx_class_list = df_input_xlsx["class"].tolist()
    #
    assert len(xlsx_names_list) == len(xlsx_class_list), "length of 'name_list' and 'class_list' misMatch"
    print(f"len(df_A_names_list) = len(df_class_list) = {len(xlsx_class_list)}\n")



    # *** Start process ***
    for i, fish_name in enumerate(xlsx_names_list):
        
        
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


        
        # *** Trim image into 2 pieces; upper for test dataset, lower for train dataset ( direction: horizontal ) ***
        half_position = int(fish.shape[0]/2)
        lower_prt_size = floor(half_position/crop_size)*crop_size # 由於設計 gen_crop_img() 時會將無法整除 crop_size 的部分從上方丟棄，
        #                                                           因此無條件捨去後再乘回 crop_size 以使 lower part 的大小與 crop_size 維持整除關係。
        cut_position = fish.shape[0] - lower_prt_size # trim 的座標是從左上算起，因此以 全高 - lower_prt_size，才是正確的切割位置。
        # print(fish.shape[0], fish.shape[1], half_position, half_position/crop_size, lower_prt_size, cut_position)
        #
        fish_upper = fish[0:cut_position, :, :]
        fish_lower = fish[cut_position:, :, :]
        # print(f"original_size: {fish.shape}, upper_size: {fish_upper.shape}, lower_size: {fish_lower.shape}\n")
        assert fish_upper.shape[0] >= fish_lower.shape[0], "Error: lower larger than upper."
        #
        #        
        # # *** Let user check the relationship between original image, upper_part, and lower_part **8
        # fish_upper_resize_to_display = cv2.resize(fish_upper, (int(fish_upper.shape[1]/5), int(fish_upper.shape[0]/5)))
        # cv2.imshow("fish_upper", fish_upper_resize_to_display)
        # fish_lower_resize_to_display = cv2.resize(fish_lower, (int(fish_lower.shape[1]/5), int(fish_lower.shape[0]/5)))
        # cv2.imshow("fish_lower", fish_lower_resize_to_display)
        # cv2.imshow("fish_resize_to_display", fish_resize_to_display)
        # cv2.waitKey(0)



        # *** Crop upper part(train dataset) and lower part(test dataset) ***
        #
        if np.random.choice([True, False], size=1, replace=False)[0]:
            used_for_test = fish_upper
            used_for_train = fish_lower
            print("test : upper, train: lower\n")
        else:
            used_for_test = fish_lower
            used_for_train = fish_upper
            print("test : lower, train: upper\n")
        #
        ## crop test
        test_crop_img_list = gen_crop_img(used_for_test, crop_size, crop_shift_region)
        test_select_img_list = drop_too_dark(test_crop_img_list, intensity, drop_ratio)
        #
        ## crop train
        train_crop_img_list = gen_crop_img(used_for_train, crop_size, crop_shift_region)
        train_select_img_list = drop_too_dark(train_crop_img_list, intensity, drop_ratio)



        # *** Save select_crop_img ***
        ## for path, e.g. ".\stacked_palmskin_tiff\20220727 CE012 palmskin_9dpf - Series002 fish 111 palmskin_9df_P_RGB only.tif"
        fish_name_list = fish_name.split(" ")
        print(fish_name_list, "\n")
        assert len(fish_name_list) == 9, "IMAGE_NAME Format Error!"
        #
        ## Extracting fish_id, e.g. "fish_1"
        fish_id = f"{fish_name_list[5]}_{fish_name_list[6]}"
        #
        ## Extracting position_tag, Anterior --> A, Posterior --> P
        position_tag = fish_name_list[-2].split("_")[2]
        assert position_tag == select_column[0].upper(), f"IMAGE_NAME ERROR, select_colum = '{select_column}', but resolve '{position_tag}'"
        #
        ## Looking up the class of current fish
        fish_size = xlsx_class_list[i]
        #
        # print(fish_size, fish_id, position_tag)
        #
        save_dir_train = f"{save_dir}/train/{fish_size}"
        save_dir_test = f"{save_dir}/test/{fish_size}"
        create_new_dir(save_dir_train, end="")
        create_new_dir(save_dir_test)
        #
        ## write cropped images
        ### train
        for i in range(len(train_select_img_list)):
            wirte_name = f"{fish_size}_{fish_id}_{position_tag}_train_{i}.tiff"
            write_path = os.path.join(save_dir_train, wirte_name)
            cv2.imwrite(write_path, train_select_img_list[i])
        ### test
        for i in range(len(test_select_img_list)):
            wirte_name = f"{fish_size}_{fish_id}_{position_tag}_test_{i}.tiff"
            write_path = os.path.join(save_dir_test, wirte_name)
            cv2.imwrite(write_path, test_select_img_list[i])



        # *** Update logs ***
        #
        ## create class_map for log count
        all_class = Counter(xlsx_class_list)
        all_class = sorted(list(all_class.keys()))
        print(all_class)
        #
        #
        ## update Current_TrainSet_log
        current_trainset_log = {
            "fish_id": fish_id,
            #
            "number_of_crop": len(train_crop_img_list),
            "number_of_drop": len(train_crop_img_list) - len(train_select_img_list),
        }
        #
        ### creat fish_size column in Current_TrainSet_log
        current_trainset_log["Class Count"] = ""
        for size in all_class: current_trainset_log[size] = 0
        #
        ### update # of saved images to Current_TrainSet_log
        current_trainset_log[fish_size] = len(train_select_img_list)
        #
        trainset_logs.append(current_trainset_log)
        ### print Current_TrainSet_log in command
        print(json.dumps(current_trainset_log, indent=2), "\n")
        #
        #
        ## update Current_TestSet_log
        current_testset_log = {
            "fish_id": fish_id,
            #
            "number_of_crop": len(test_crop_img_list),
            "number_of_drop": len(test_crop_img_list) - len(test_select_img_list),
        }
        #
        ### creat fish_size column in Current_TestSet_log
        current_testset_log["Class Count"] = ""
        for size in all_class: current_testset_log[size] = 0
        #
        ### update # of saved images to Current_TestSet_log
        current_testset_log[fish_size] = len(test_select_img_list)
        #
        testset_logs.append(current_testset_log)
        ### print Current_TestSet_log in command
        print(json.dumps(current_testset_log, indent=2), "\n")



    # *** Change logs into Dataframe, Save and Show in command ***
    #
    ## get time to as file name
    time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S') # get time to as file name
    #
    #
    #
    ## trainSet_log
    df_trainset_logs = pd.DataFrame(trainset_logs)
    df_trainset_logs.loc['TOTAL'] = df_trainset_logs.select_dtypes(np.number).sum()
    print("="*100, "\n")
    print("Train Set:\n\n", df_trainset_logs, "\n")
    #
    trainset_logs_path_abs = f"{save_dir}/[log_train]{time_stamp}_using_(mk_dataset_horiz_cut).json"
    df_trainset_logs.to_json(trainset_logs_path_abs, orient ='index', indent=2)
    print("\n", f"Train Set log save @ \n-> {trainset_logs_path_abs}\n")
    #
    #
    #
    ## testSet_log
    df_testset_logs = pd.DataFrame(testset_logs)
    df_testset_logs.loc['TOTAL'] = df_testset_logs.select_dtypes(np.number).sum()
    print("="*100, "\n")
    print("Test Set:\n\n",df_testset_logs, "\n")
    #
    testset_logs_path_abs = f"{save_dir}/[log_test]{time_stamp}_using_(mk_dataset_horiz_cut).json"
    df_testset_logs.to_json(testset_logs_path_abs, orient ='index', indent=2)
    print("\n", f"Test Set log save @ \n-> {testset_logs_path_abs}\n")



    print("="*100, "\n", "process all complete !", "\n")