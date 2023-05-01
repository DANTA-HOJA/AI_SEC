import os
import sys
import argparse
from datetime import datetime
from collections import Counter
from math import floor
import json
import toml
import tomlkit # use this branch can preserve all content and order in the `toml` file

from tqdm.auto import tqdm
import cv2
import numpy as np
import pandas as pd

sys.path.append("/home/rime97410000/ZebraFish_Code/ZebraFish_AP_POS/modules") # add path to scan customized module
from fileop import create_new_dir
from dataop import get_fish_ID_pos
from datasetop import get_args, gen_dataset_param_name, gen_crop_img, drop_too_dark, save_crop_img, \
                      append_log, save_dataset_logs, gen_train_selected_summary, save_dataset_config, \
                      save_dark_ratio_log



if __name__ == "__main__":
    
    
    with open("mk_dataset_horiz_cut.toml", mode="r") as f_reader:
        config = toml.load(f_reader)


    # *** Variable ***
    ## set vars from config file (.toml)
    script_name = config["script_name"]
    data_root  = os.path.normpath(config["data"]["root"])
    xlsx_file  = config["data"]["brightfield"]["xlsx_file"]
    sheet_name = config["data"]["brightfield"]["sheet_name"]
    palmskin_desc       = config["data"]["stacked_palmskin"]["desc"]
    palmskin_result_key = config["data"]["stacked_palmskin"]["result_key"]
    crop_size    = config["gen_param"]["crop_size"]
    shift_region = config["gen_param"]["shift_region"]
    intensity    = config["gen_param"]["intensity"]
    drop_ratio   = config["gen_param"]["drop_ratio"]
    random_seed  = config["gen_param"]["random_seed"]
    dataset_root = os.path.normpath(config["dataset"]["root"])
    ## compose/extract vars
    data_name = data_root.split(os.sep)[-1]
    xlsx_file_path = os.path.join(data_root, r"{Modify}_xlsx", xlsx_file)
    stacked_palmskin_dir   = os.path.join(data_root, f"{{{palmskin_desc}}}_RGB_reCollection", palmskin_result_key)
    np.random.seed(random_seed)
    dataset_param_name = gen_dataset_param_name(xlsx_file, crop_size, shift_region, intensity, drop_ratio, random_seed)
    save_dir_A_only = os.path.join(dataset_root, data_name, "fish_dataset_horiz_cut_1l2_A_only", sheet_name, dataset_param_name)
    save_dir_P_only = os.path.join(dataset_root, data_name, "fish_dataset_horiz_cut_1l2_P_only", sheet_name, dataset_param_name)
    save_dir_Mix_AP = os.path.join(dataset_root, data_name, "fish_dataset_horiz_cut_1l2_Mix_AP", sheet_name, dataset_param_name)
    print("")
    create_new_dir(save_dir_A_only)
    create_new_dir(save_dir_P_only)
    create_new_dir(save_dir_Mix_AP)



    # *** Load Excel sheet as DataFrame(pandas) ***
    df_input_xlsx :pd.DataFrame = pd.read_excel(xlsx_file_path, engine = 'openpyxl', sheet_name=sheet_name)
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
        train_darkratio_log = {}
        test_darkratio_log = {}
        
        
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
        for i, fish_name_for_data in enumerate(df_palmskin_list):
            
            
            # *** Load image ***
            fish_path = os.path.normpath(f"{stacked_palmskin_dir}/{fish_name_for_data}.tif")
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
            part_of_train = ""
            part_of_test = ""
            if np.random.choice([True, False], size=1, replace=False)[0]:
                used_for_test = fish_upper
                used_for_train = fish_lower
                part_of_test = "U"
                part_of_train = "L"
                rand_choice_result["test : upper, train: lower"] += 1
            else:
                used_for_test = fish_lower
                used_for_train = fish_upper
                part_of_test = "L"
                part_of_train = "U"
                rand_choice_result["test : lower, train: upper"] += 1
            #
            ## crop test
            test_crop_img_list = gen_crop_img(used_for_test, crop_size, shift_region)
            test_select_crop_img_list, test_drop_crop_img_list = drop_too_dark(test_crop_img_list, intensity, drop_ratio)
            #
            ## crop train
            train_crop_img_list = gen_crop_img(used_for_train, crop_size, shift_region)
            train_select_crop_img_list, train_drop_crop_img_list = drop_too_dark(train_crop_img_list, intensity, drop_ratio)



            # *** Extracting / Looking up the information on current fish ***
            ## path, e.g. "...\{*}_RGB_reCollection\[*result]\20220727_CE012_palmskin_9dpf - Series002_fish_111_P_RGB.tif"
            fish_id, fish_pos = get_fish_ID_pos(fish_path)
            #
            ## looking up the class of current fish
            fish_size = df_class_list[i]
            #
            # print(fish_size, fish_id, fish_pos)
            fish_name_for_dataset = f"{fish_size}_fish_{fish_id}_{fish_pos}"
            pbar_n_fish.desc = f"Cropping {pos}... '{fish_name_for_dataset}' "
            pbar_n_fish.refresh()



            for dir_path in [save_dir, save_dir_Mix_AP]:
                
                # save 'test_select_crop_img_list' 
                save_crop_img_kwargs = {
                    "save_dir"      : os.path.join(dir_path, "test"),
                    "crop_img_list" : test_select_crop_img_list,
                    "darkratio_log" : test_darkratio_log,
                    "crop_img_desc" : "selected",
                    "fish_size"     : fish_size,
                    "fish_id"       : fish_id,
                    "fish_pos"      : fish_pos,
                    "tqdm_process"  : pbar_n_select,
                    "tqdm_overwrite_desc" : "Saving selected... Test "
                }
                save_crop_img(**save_crop_img_kwargs)
                
                # save 'test_drop_crop_img_list' 
                save_crop_img_kwargs = {
                    "save_dir"      : os.path.join(dir_path, "test"),
                    "crop_img_list" : test_drop_crop_img_list,
                    "darkratio_log" : test_darkratio_log,
                    "crop_img_desc" : "drop",
                    "fish_size"     : fish_size,
                    "fish_id"       : fish_id,
                    "fish_pos"      : fish_pos,
                    "tqdm_process"  : pbar_n_drop,
                    "tqdm_overwrite_desc" : "Saving drop... Test "
                }
                save_crop_img(**save_crop_img_kwargs)

                # save 'train_select_crop_img_list' 
                save_crop_img_kwargs = {
                    "save_dir"      : os.path.join(dir_path, "train"),
                    "crop_img_list" : train_select_crop_img_list,
                    "darkratio_log" : train_darkratio_log,
                    "crop_img_desc" : "selected",
                    "fish_size"     : fish_size,
                    "fish_id"       : fish_id,
                    "fish_pos"      : fish_pos,
                    "tqdm_process"  : pbar_n_select,
                    "tqdm_overwrite_desc" : "Saving selected... Train "
                }
                save_crop_img(**save_crop_img_kwargs)

                # save 'train_drop_crop_img_list' 
                save_crop_img_kwargs = {
                    "save_dir"      : os.path.join(dir_path, "train"),
                    "crop_img_list" : train_drop_crop_img_list,
                    "darkratio_log" : train_darkratio_log,
                    "crop_img_desc" : "drop",
                    "fish_size"     : fish_size,
                    "fish_id"       : fish_id,
                    "fish_pos"      : fish_pos,
                    "tqdm_process"  : pbar_n_drop,
                    "tqdm_overwrite_desc" : "Saving drop... Train "
                }
                save_crop_img(**save_crop_img_kwargs)



            # *** Update 'trainset_logs' ***
            append_log_kwargs = {
                "logs"                 : trainset_logs,
                "fish_size"            : fish_size,
                "fish_id"              : fish_id,
                "fish_pos"             : fish_pos,
                "selected_part"        : part_of_train,
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
                "fish_id"              : fish_id,
                "fish_pos"             : fish_pos,
                "selected_part"        : part_of_test,
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

        
        print("\n", json.dumps(rand_choice_result, indent=4))
        

        # *** Save logs into XLSX and show in CLI ***
        ## get time to as file name
        time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        for dir_path in [save_dir, save_dir_Mix_AP]:
            
            if dir_path == save_dir_Mix_AP: show_df = False
            else : show_df = True
            
            # 'trainset_logs'
            save_dataset_logs_kwargs = {
                "logs"        : trainset_logs,
                "save_dir"    : dir_path,
                "log_desc"    : f"Logs_{pos[0]}_train",
                "script_name" : script_name,
                "CLI_desc"    : "Train :\n",
                "time_stamp"  : time_stamp,
                "show_df"     : show_df
            }
            save_dataset_logs(**save_dataset_logs_kwargs)
            save_dark_ratio_log(train_darkratio_log, dir_path, f"Logs_{pos[0]}_train")

            # 'testset_logs'
            save_dataset_logs_kwargs = {
                "logs"        : testset_logs,
                "save_dir"    : dir_path,
                "log_desc"    : f"Logs_{pos[0]}_test",
                "script_name" : script_name,
                "CLI_desc"    : "Test :\n",
                "time_stamp"  : time_stamp,
                "show_df"     : show_df
            }
            save_dataset_logs(**save_dataset_logs_kwargs)
            save_dark_ratio_log(test_darkratio_log, dir_path, f"Logs_{pos[0]}_test")


    # Generate '{Logs}_train_selected_summary.log', 'dataset_config.yaml'
    for dir_path in [save_dir_A_only, save_dir_P_only, save_dir_Mix_AP]:
        gen_train_selected_summary(dir_path, all_class)
        save_dataset_config(dir_path, config)
    
    
    print("="*100, "\n", "process all complete !", "\n")