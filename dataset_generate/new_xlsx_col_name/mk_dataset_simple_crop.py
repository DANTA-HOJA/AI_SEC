import os
import sys
import argparse
from datetime import datetime
from collections import Counter

from tqdm.auto import tqdm
import pandas as pd
import cv2

sys.path.append("/home/rime97410000/ZebraFish_Code/ZebraFish_AP_POS/modules") # add path to scan customized module
from fileop import create_new_dir
from dataop import get_fish_ID_pos
from dataset.utils import gen_dataset_name, gen_crop_img, drop_too_dark, save_crop_img, append_log, save_dataset_logs



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
    df_class_list = df_input_xlsx["class"].tolist()
    ## get how many classes in xlsx
    all_class = Counter(df_class_list)
    all_class = sorted(list(all_class.keys()))
    # print(all_class)



    for key, value in train_and_test.items():

        # variable
        logs = []


        # *** Print CMD section divider ***
        print("="*100, "\n")
        print(f"{key} : {value}\n")


        if value == "A" : df_palmskin_list = df_input_xlsx["Anterior (SP8, .tif)" ].tolist()
        if value == "P" : df_palmskin_list = df_input_xlsx["Posterior (SP8, .tif)"].tolist()
        assert len(df_palmskin_list) == len(df_class_list), "length of 'palmskin_list' and 'class_list' misMatch"
        print(f"len(df_palmskin_list) = len(df_class_list) = {len(df_class_list)}\n")


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
            ## check orient by user
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
            fish_name_comb = f'{fish_size}_fish_{fish_ID}_{fish_pos}'
            pbar_n_fish.desc = f"Cropping {key}({value})... '{fish_name_comb}' "
            pbar_n_fish.refresh()


            # *** Save 'select_crop_img' ***
            save_crop_img_kwargs = {
                "save_dir"      : os.path.join(save_dir, key),
                "crop_img_list" : select_crop_img_list,
                "crop_img_desc" : "selected",
                "fish_size"     : fish_size,
                "fish_id"       : fish_ID,
                "fish_pos"      : fish_pos,
                "tqdm_process"  : pbar_n_select,
            }
            save_crop_img(**save_crop_img_kwargs)


            # *** Save 'drop_crop_img' ***
            save_crop_img_kwargs = {
                "save_dir"      : os.path.join(save_dir, key),
                "crop_img_list" : drop_crop_img_list,
                "crop_img_desc" : "drop",
                "fish_size"     : fish_size,
                "fish_id"       : fish_ID,
                "fish_pos"      : fish_pos,
                "tqdm_process"  : pbar_n_drop,
            }
            save_crop_img(**save_crop_img_kwargs)


            # *** Update log of current fish ***
            append_log_kwargs = {
                "logs"                 : logs,
                "fish_size"            : fish_size,
                "fish_id"              : fish_ID,
                "fish_pos"             : fish_pos,
                "all_class"            : all_class,
                "crop_img_list"        : crop_img_list, 
                "select_crop_img_list" : select_crop_img_list, 
                "drop_crop_img_list"   : drop_crop_img_list
            }
            append_log(**append_log_kwargs)


            pbar_n_fish.update(1)
            pbar_n_fish.refresh()
        
        pbar_n_fish.close()
        pbar_n_select.close()
        pbar_n_drop.close()
        
        
        # *** Save logs into XLSX and show in CLI ***
        ## get time to as file name
        time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        save_dataset_logs_kwargs = {
            "logs"        : logs,
            "save_dir"    : save_dir,
            "log_desc"    : f"Logs_{value}_{key}",
            "script_name" : "mk_dataset_simple_crop",
            "time_stamp"  : time_stamp
        }
        save_dataset_logs(**save_dataset_logs_kwargs)


    print("="*100, "\n", "process all complete !", "\n")