import os
import sys
import argparse
from datetime import datetime
from collections import Counter

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import cv2

sys.path.append(r"C:\Users\confocal_microscope\Desktop\ZebraFish_AP_POS\modules") # add path to scan customized module
from fileop import create_new_dir
from norm_name import get_fish_ID_pos
from dataset_generate import gen_dataset_name, gen_crop_img, drop_too_dark


# *** show images methods ***
# cv2.imshow("fish", fish)
# cv2.waitKey(0)
# plt_show.img_rgb("fish", fish)
# plt_show.img_gray("fish", fish)
# plt_show.img_by_channel("fish", fish)



def get_args():
    
    parser = argparse.ArgumentParser(description="zebrafish project: crop images into small pieces")
    parser.add_argument(
        "--ap_data_root",
        type=str,
        required=True,
        help="The root path of the data.",
    )
    parser.add_argument(
        "--xlsx_file",
        type=str,
        required=True,
        help=r"The name of the Excel book, under 'ap_data_root/{Modify}_xlsx' ",
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
        help="The folder storing the 'stacked palmskin' images, relative to 'ap_data_root' ",
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
        "--dataset_root",
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
    ap_data_root = args.ap_data_root
    data_name = ap_data_root.split(os.sep)[-1]
    #
    xlsx_file = args.xlsx_file
    xlsx_file_fullpath = os.path.join(ap_data_root, r"{Modify}_xlsx", xlsx_file)
    sheet_name = args.sheet_name
    #
    preprocess_method_desc = "ch4_min_proj, outer_rect"
    stacked_palmskin_dir = os.path.join(ap_data_root, f"{{{preprocess_method_desc}}}_RGB_reCollection", args.stacked_palmskin_dir)
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
    dataset_root = args.dataset_root
    gen_name = gen_dataset_name(xlsx_file, crop_size, crop_shift_region, intensity, drop_ratio)
    save_dir = os.path.join(args.dataset_root, data_name, f"fish_dataset_simple_crop_{train_and_test_no_slash}", gen_name)
    print("")
    create_new_dir(save_dir)



    # *** Load Excel sheet as DataFrame(pandas) ***
    df_input_xlsx :pd.DataFrame = pd.read_excel(xlsx_file_fullpath, engine = 'openpyxl', sheet_name=sheet_name)
    # print(df_input_xlsx)
    
    
    for key, value in train_and_test.items():

        # variable
        logs = []


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
        pbar_n_select = tqdm(desc="Saving selected... ")
        pbar_n_drop = tqdm(desc="Saving drop... ")
        
        
        # *** Start process ***
        for i, fish_name in enumerate(df_palmskin_list):
            
            
            # *** Load image ***
            fish_path = os.path.normpath(f"{stacked_palmskin_dir}/{fish_name}.tif")
            fish = cv2.imread(fish_path)
            
            
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
            select_crop_img_list, drop_crop_img_list = drop_too_dark(crop_img_list, intensity, drop_ratio)


            # *** Extracting / Looking up the information on current fish ***
            ## path, e.g. "...\{*}_RGB_reCollection\[*result]\20220727_CE012_palmskin_9dpf - Series002_fish_111_P_RGB.tif"
            fish_ID, fish_pos = get_fish_ID_pos(fish_path)
            #
            ## looking up the class of current fish
            fish_size = df_class_list[i]
            #
            # print(fish_size, fish_ID, fish_pos)
            fish_name_comb = f'{fish_size}_{fish_ID}_{fish_pos}'
            pbar_n_fish.desc  = f"Cropping {key}({value})... '{fish_name_comb}' "
            pbar_n_fish.refresh()
            
            
            # *** Save 'select_crop_img' ***
            ## adjust settings of 'pbar_n_select'
            pbar_n_select.n     = 0   # current value
            pbar_n_select.total = len(select_crop_img_list)
            pbar_n_select.refresh()
            #
            ## write selected images
            for j in range(len(select_crop_img_list)):
                save_dir_size = os.path.join(save_dir_set, "selected", fish_size)
                create_new_dir(save_dir_size, use_tqdm=True)
                write_name = f"{fish_name_comb}_crop_{j}.tiff"
                write_path = os.path.join(save_dir_size, write_name)

                select_crop_img = select_crop_img_list[j] # convenience to debug preview
                cv2.imwrite(write_path, select_crop_img)
                # cv2.imshow(write_name, select_crop_img)
                # cv2.waitKey(0)
                
                pbar_n_select.update(1)
                pbar_n_select.refresh()


            # *** Save 'drop_crop_img' ***
            ## adjust settings of 'pbar_n_drop'
            pbar_n_drop.n     = 0   # current value
            pbar_n_drop.total = len(drop_crop_img_list)
            pbar_n_drop.refresh()
            #
            ## write drop images
            for j in range(len(drop_crop_img_list)):
                save_dir_size = os.path.join(save_dir_set, "drop", fish_size)
                create_new_dir(save_dir_size, use_tqdm=True)
                write_name = f"{fish_name_comb}_crop_{j}.tiff"
                write_path = os.path.join(save_dir_size, write_name)

                drop_crop_img = drop_crop_img_list[j] # convenience to debug preview
                cv2.imwrite(write_path, drop_crop_img)
                # cv2.imshow(write_name, drop_crop_img)
                # cv2.waitKey(0)
                
                pbar_n_drop.update(1)
                pbar_n_drop.refresh()


            # *** Update log of current fish ***
            current_log = {
                "fish_name_comb": fish_name_comb,
                #
                "number_of_crop": len(crop_img_list),
                "number_of_drop": len(drop_crop_img_list),
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
        pbar_n_select.close()
        pbar_n_drop.close()
        
        
        # *** Change logs into Dataframe and show in command ***
        df = pd.DataFrame(logs)
        df.loc['TOTAL'] = df.select_dtypes(np.number).sum()
        # print("="*100, "\n")
        print("\n\n", df, "\n")
        
        
        # *** Save logs ***
        ## get time to as file name
        time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S') # get time to as file name
        log_path_abs = f"{save_dir}/{{log_{value}_{key}}}_{time_stamp}_using_(mk_dataset_simple_crop).xlsx"
        df.to_excel(log_path_abs, engine="openpyxl")
        print("\n", f"log save @ \n-> {log_path_abs}\n")


    print("="*100, "\n", "process all complete !", "\n")