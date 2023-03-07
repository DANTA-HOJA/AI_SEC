import os
import sys
import argparse
from datetime import datetime
from collections import Counter

from math import floor
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import cv2

sys.path.append(r"C:\Users\confocal_microscope\Desktop\ZebraFish_AP_POS\modules") # add path to scan customized module
from fileop import create_new_dir
from norm_name import get_fish_ID_pos
from dataset_generate import gen_dataset_name, gen_crop_img, drop_too_dark, crop_img_saver, append_log, logs_saver


# *** Show images methods ***
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
        "--random_seed",
        type=int,
        default=2022,
        help="random seed, for choosing upper or lower as train part.",
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
    random_seed = args.random_seed # for choosing upper or lower as train part.
    np.random.seed(random_seed)
    #
    dataset_root = args.dataset_root
    gen_name = gen_dataset_name(xlsx_file, crop_size, crop_shift_region, intensity, drop_ratio, random_seed)
    save_dir_A_only = os.path.join(args.dataset_root, data_name, "fish_dataset_horiz_cut_1l2_A_only", gen_name)
    save_dir_P_only = os.path.join(args.dataset_root, data_name, "fish_dataset_horiz_cut_1l2_P_only", gen_name)
    save_dir_Mix_AP = os.path.join(args.dataset_root, data_name, "fish_dataset_horiz_cut_1l2_Mix_AP", gen_name)
    print("")
    create_new_dir(save_dir_A_only)
    create_new_dir(save_dir_P_only)
    create_new_dir(save_dir_Mix_AP)



    # *** Load Excel sheet as DataFrame(pandas) ***
    df_input_xlsx :pd.DataFrame = pd.read_excel(xlsx_file_fullpath, engine = 'openpyxl', sheet_name=sheet_name)
    # print(df_input_xlsx)
    df_class_list = df_input_xlsx["class"].tolist()
    ## get how many classes in xlsx
    all_class = Counter(df_class_list)
    all_class = sorted(list(all_class.keys()))
    # print(all_class)



    pos_dict = {"Anterior": save_dir_A_only, "Posterior": save_dir_P_only}
    for pos, save_dir in pos_dict.items():
        
        # variable
        trainset_logs = []
        testset_logs = []
        rand_choice_result = {"test : upper, train: lower": 0, 
                              "test : lower, train: upper": 0}
        
        
        # *** Print CMD section divider ***
        print("="*100, "\n")
        print(f"Generating {pos}... \n")


        df_palmskin_list = df_input_xlsx[f"{pos} (SP8, .tif)"].tolist()
        assert len(df_palmskin_list) == len(df_class_list), "length of 'palmskin_list' and 'class_list' misMatch"
        print(f"len(df_palmskin_list) = len(df_class_list) = {len(df_class_list)}\n")


        # *** Create progress bar ***
        pbar_n_fish = tqdm(total=len(df_palmskin_list), desc=f"Cropping {pos}... ")
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
            # # *** Let user check the relationship between original image, upper_part, and lower_part ***
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
                rand_choice_result["test : upper, train: lower"] += 1
            else:
                used_for_test = fish_lower
                used_for_train = fish_upper
                rand_choice_result["test : lower, train: upper"] += 1
            #
            ## crop test
            test_crop_img_list = gen_crop_img(used_for_test, crop_size, crop_shift_region)
            test_select_crop_img_list, test_drop_crop_img_list = drop_too_dark(test_crop_img_list, intensity, drop_ratio)
            #
            ## crop train
            train_crop_img_list = gen_crop_img(used_for_train, crop_size, crop_shift_region)
            train_select_crop_img_list, train_drop_crop_img_list = drop_too_dark(train_crop_img_list, intensity, drop_ratio)



            # *** Extracting / Looking up the information on current fish ***
            ## path, e.g. "...\{*}_RGB_reCollection\[*result]\20220727_CE012_palmskin_9dpf - Series002_fish_111_P_RGB.tif"
            fish_ID, fish_pos = get_fish_ID_pos(fish_path)
            #
            ## looking up the class of current fish
            fish_size = df_class_list[i]
            #
            # print(fish_size, fish_ID, fish_pos)
            fish_name_comb = f'{fish_size}_fish_{fish_ID}_{fish_pos}'
            pbar_n_fish.desc = f"Cropping {pos}... '{fish_name_comb}' "
            pbar_n_fish.refresh()



            for dir_path in [save_dir, save_dir_Mix_AP]:
                
                # save 'test_select_crop_img_list' 
                crop_img_saver_kwargs = {
                    "save_dir"      : os.path.join(dir_path, "test"),
                    "crop_img_list" : test_select_crop_img_list,
                    "crop_img_desc" : "selected",
                    "fish_size"     : fish_size,
                    "fish_id"       : fish_ID,
                    "fish_pos"      : fish_pos,
                    "tqdm_process"  : pbar_n_select,
                    "tqdm_overwrite_desc" : "Saving selected... Test "
                }
                crop_img_saver(**crop_img_saver_kwargs)
                
                # save 'test_drop_crop_img_list' 
                crop_img_saver_kwargs = {
                    "save_dir"      : os.path.join(dir_path, "test"),
                    "crop_img_list" : test_drop_crop_img_list,
                    "crop_img_desc" : "drop",
                    "fish_size"     : fish_size,
                    "fish_id"       : fish_ID,
                    "fish_pos"      : fish_pos,
                    "tqdm_process"  : pbar_n_drop,
                    "tqdm_overwrite_desc" : "Saving drop... Test "
                }
                crop_img_saver(**crop_img_saver_kwargs)

                # save 'train_select_crop_img_list' 
                crop_img_saver_kwargs = {
                    "save_dir"      : os.path.join(dir_path, "train"),
                    "crop_img_list" : train_select_crop_img_list,
                    "crop_img_desc" : "selected",
                    "fish_size"     : fish_size,
                    "fish_id"       : fish_ID,
                    "fish_pos"      : fish_pos,
                    "tqdm_process"  : pbar_n_select,
                    "tqdm_overwrite_desc" : "Saving selected... Train "
                }
                crop_img_saver(**crop_img_saver_kwargs)

                # save 'train_drop_crop_img_list' 
                crop_img_saver_kwargs = {
                    "save_dir"      : os.path.join(dir_path, "train"),
                    "crop_img_list" : train_drop_crop_img_list,
                    "crop_img_desc" : "drop",
                    "fish_size"     : fish_size,
                    "fish_id"       : fish_ID,
                    "fish_pos"      : fish_pos,
                    "tqdm_process"  : pbar_n_drop,
                    "tqdm_overwrite_desc" : "Saving drop... Train "
                }
                crop_img_saver(**crop_img_saver_kwargs)



            # *** Update 'trainset_logs' ***
            append_log_kwargs = {
                "logs"                 : trainset_logs,
                "fish_size"            : fish_size,
                "fish_id"              : fish_ID,
                "fish_pos"             : fish_pos,
                "all_class"            : all_class,
                "crop_img_list"        : train_crop_img_list, 
                "select_crop_img_list" : train_select_crop_img_list, 
                "drop_crop_img_list"   : train_drop_crop_img_list
            }
            append_log(**append_log_kwargs)

            # *** Update 'testset_logs' ***
            append_log_kwargs = {
                "logs"                 : testset_logs,
                "fish_size"            : fish_size,
                "fish_id"              : fish_ID,
                "fish_pos"             : fish_pos,
                "all_class"            : all_class,
                "crop_img_list"        : test_crop_img_list, 
                "select_crop_img_list" : test_select_crop_img_list, 
                "drop_crop_img_list"   : test_drop_crop_img_list
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
        for dir_path in [save_dir, save_dir_Mix_AP]:
            
            if dir_path == save_dir_Mix_AP: show_df = False
            else : show_df = True
            
            # 'trainset_logs'
            logs_saver_kwargs = {
                "logs"        : trainset_logs,
                "save_dir"    : dir_path,
                "log_desc"    : f"Logs_{pos[0]}_train",
                "script_name" : "mk_dataset_horiz_cut",
                "CLI_desc"    : "Train :\n",
                "time_stamp"  : time_stamp,
                "show_df"     : show_df
            }
            logs_saver(**logs_saver_kwargs)

            # 'testset_logs'
            logs_saver_kwargs = {
                "logs"        : testset_logs,
                "save_dir"    : dir_path,
                "log_desc"    : f"Logs_{pos[0]}_test",
                "script_name" : "mk_dataset_horiz_cut",
                "CLI_desc"    : "Test :\n",
                "time_stamp"  : time_stamp,
                "show_df"     : show_df
            }
            logs_saver(**logs_saver_kwargs)


    print("="*100, "\n", "process all complete !", "\n")