import os
import sys
import re
from typing import List, Dict, Union, Tuple
from colorama import Fore, Style
from pathlib import Path

import json
import toml
import pandas as pd
from logging import Logger

# -----------------------------------------------------------------------------------

#  INFO:  gallery_size: 256 (8GB), 512 (4GB) for both type of gallery

#  INFO: 
# `train_config` parser `stdev` -> `classif_strategy`, e.g. 0.75_STDEV -> 075STDEV
#   --> dataset_config 只有更改為 "只需要 xlsx_file"，cropping 的動作都沒有修改

# WARNING: `dataset_config` 多一個 key `sheet_name` ( 不是很重要了， `dataset_config` 的資訊都可以透過 `train_config` 重組)

# WARNING: `history_dir`: `{Logs}_train.xlsx` + `{Logs}_valid.xlsx` = `{Logs}_training_log.xlsx`

# WARNING: `training_trend` 有三種 : 1. `training_trend_average_f1.png`
#                                         2. `training_trend_weighted_f1.png` 
#                                         3. `training_trend_maweavg_f1.png`

#  TODO:  data_operate 要新增 config 紀錄 parameters（原本 parameters 有存在 Log 內），並 parse 進來

#  TODO: 
# 因為要新增 `不同 image 作為 source` 的功能，資料夾路徑會加上 result_alias (result_key), palmskin_desc 層級，
# 新的 configs 都要加上 result_alias, palmskin_desc ( KEY_NAME 待定 )，
#   --> cropping, training 的 scripts, configs 也要加上路徑
# 舊的 train_config 找不到 `result_alias` 的話自動將 value 設為 "( RGB_HE_mix )"
# 舊的 train_config 找不到 `palmskin_desc` 的話自動將 value 設為 "( ch4_min_proj, outer_rect )"
# TBA: 考慮把 desc 換成 uuid

#  TODO:  透過 history name 判斷哪些還沒更新過，是 `更新` 不是 Rewrite (保存可能使用 PKL or XLSX)

#  TODO: 
# 把 dataset 和 data 的資訊 parser 進來，檢查重複性，若有重複要刪除，
# 例如："(TrainSet)" : "{ dataset_* }_{ train_* }_{ valid_* }" 在 dataset info 加進來之後應該要刪除

#  TODO: 
# 產生專門 render 路徑的 script
# data, dataset 的路徑都可以由 train_config 重新組合，
# 查巡當下的電腦裡面是否有對應的 data, dataset 存在，有對應的檔案即可替原始的 marker 加上 HYPERLINK
# 生成的 xlsx 在每台 PC 上要重新 render 路徑

#  TODO:  產生 HYPERLINE text 的 funcion

# -----------------------------------------------------------------------------------



class SinglePredictionParser():

    def __init__(self, prediction_dir:Path, log:Logger) -> None:
        
        # -----------------------------------------------------------------------------------
        if isinstance(prediction_dir, Path): 
            self.prediction_dir = prediction_dir
        else: raise TypeError(f"`prediction_dir` should be a `Path` object, please using `from pathlib import Path`")
        
        assert self.prediction_dir.exists(), f"Can't reach the folder '{self.prediction_dir}'"
        
        self.prediction_dir_split = str(prediction_dir).split(os.sep)
        self.prediction_dir_name = self.prediction_dir_split[-1] # e.g. "20230426_05_50_33_{Tested_PredByFish_CAM}_{69_epochs_AugOnFly}_{best}_{maweavg_f1_0.83707}"
        self.model_prediction_dir_idx = None
        self.find_model_prediction_dir_in_path()

        self.zebrafish_db_root = Path(os.sep.join(self.prediction_dir_split[:self.model_prediction_dir_idx]))
        self.data_processed_dir = self.zebrafish_db_root.joinpath(r"{Data}_Preprocessed")
        self.dataset_cropped_dir = self.zebrafish_db_root.joinpath(r"{Dataset}_Cropped")
        self.model_cmd_dir = self.zebrafish_db_root.joinpath(r"{Model}_CMD")
        self.model_prediction_dir = self.zebrafish_db_root.joinpath(r"{Model}_Prediction")
        
        self.parsed_dict = { "History Name": self.prediction_dir_name }
        self.find_files_cnt = 0
        self.parsed_df = None
        
        self.prediction_scan_items = {
            "(TrainConfig)" : "train_config.toml", # config
            # -----------------------------------------------------------------------------------
            "(TrainSet)" : "{ dataset_* }_{ train_* }_{ valid_* }",
            "(TestSet)"  : "{ datatest_* }_{ test_* }",
            # -----------------------------------------------------------------------------------
            "(Training) Time" : "{ training time }_{ * sec }",
            "(Training) Train_Log"      : r"{Logs}_train.xlsx", # (1 in 2 files) old training log
            "(Training) Valid_Log"      : r"{Logs}_valid.xlsx", # (2 in 2 files) old training log
            "(Training) Training_Log"   : r"{Logs}_training_log.xlsx", # (1 files) new training log (train + valid)
            "(Training) Best_Valid_Log" : r"{Logs}_best_valid.log",
            "(Training) Best_Model"  : "best_model.pth",
            "(Training) Final_Model" : "final_model.pth",
            "(TrainFigure) Average_f1"  : "training_trend_average_f1.png", # (old 1) training_trend, average_f1 = (micro + macro)/2
            "(TrainFigure) Weighted_f1" : "training_trend_weighted_f1.png", # (old 2) training_trend
            "(TrainFigure) Maweavg_f1"  : "training_trend_maweavg_f1.png", # (new) training_trend, maweavg_f1 = (macro + weighted)/2
            # -----------------------------------------------------------------------------------
            "(PredByImg) Report" : "{Report}_PredByImg.log",
            "(PredByImg) Score"  : "{Logs}_PredByImg_maweavg_f1_*.toml",
            # -----------------------------------------------------------------------------------
            "(PredByFish) Report"      : "{Report}_PredByFish.log",
            "(PredByFish) Predict_Ans" : "{Logs}_PredByFish_predict_ans.log",
            "(PredByFish) Score"       : "{Logs}_PredByFish_maweavg_f1_*.toml",
            # -----------------------------------------------------------------------------------
            "(Cam) Result_Dir"  : "cam_result",
            "(Cam) Gallery_Dir" : "!--- CAM Gallery",
        }
        
        self.log = log
        self.log_key_align = "26"
        self.file_state_marker = {"found": "\u2713", # "\u2713" == ✓
                                  "not_found": "\u2717"} # "\u2717" == ✗
        self.empty_cell_marker = "---"
        # -----------------------------------------------------------------------------------
        
        
    def find_model_prediction_dir_in_path(self): # To find `{Model}_Prediction` in `split_path`
        for i, text in enumerate(self.prediction_dir_split):
            if r"{Model}_Prediction" in text: self.model_prediction_dir_idx = i
        # -----------------------------------------------------------------------------------


    def scan_file(self, key:str): # Scan file exists and output msg to CLI
            path_list = list(self.prediction_dir.glob(self.prediction_scan_items[key]))

            if len(path_list) == 1:
                self.find_files_cnt += 1
                target_name = str(path_list[0]).split(os.sep)[-1]
                self.log.info(f"{key:{self.log_key_align}}: [ {self.file_state_marker['found']} ] {target_name}")
                return path_list[0], target_name
            elif len(path_list) > 1:
                raise ValueError(f"'{self.prediction_scan_items[key]}' should be a unique file in each prediction folder, "
                                 f"but found {len(path_list)} similar files")
            else:
                self.log.info((f"{Fore.BLACK}{key:{self.log_key_align}}: "
                               f"[ {self.file_state_marker['not_found']} ] {self.prediction_scan_items[key]}{Style.RESET_ALL}"))
                return None
            # -----------------------------------------------------------------------------------


    def parse_train_config(self): # For "(TrainConfig)" : "train_config.toml" 
        key = "(TrainConfig)"
        target = self.scan_file(key)
        if target is not None:
            path, target_name = target
            with open(path, mode="r") as f_reader: train_config = toml.load(f_reader)
            # dataset
            self.parsed_dict["(TrainConfig) dataset.name"] = train_config["dataset"]["name"]
            try: # self.parsed_dict["(TrainConfig) dataset.palmskin_desc"], new folder hierarchy
                self.parsed_dict["(TrainConfig) dataset.palmskin_desc"] = train_config["dataset"]["palmskin_desc"]
            except KeyError: # old hierarchy --> default "( ch4_min_proj, outer_rect )"
                self.parsed_dict["(TrainConfig) dataset.palmskin_desc"] = "( ch4_min_proj, outer_rect )"
            self.parsed_dict["(TrainConfig) dataset.gen_method"] = train_config["dataset"]["gen_method"]
            try: # self.parsed_dict["(TrainConfig) dataset.result_alias"], new folder hierarchy
                self.parsed_dict["(TrainConfig) dataset.result_alias"] = train_config["dataset"]["result_alias"]
            except KeyError: # old hierarchy --> default "( RGB_HE_mix )"
                self.parsed_dict["(TrainConfig) dataset.result_alias"] = "( RGB_HE_mix )"
            try: # self.parsed_dict["(TrainConfig) dataset.gen_method"], key == 'stdev' (old key)
                stdev = str(train_config["dataset"]["stdev"]) # e.g. "0.75_STDEV"
                stdev = float(stdev.replace("_STDEV", ""))*100 # e.g. 0.75
                self.parsed_dict["(TrainConfig) dataset.classif_strategy"] = f"{int(stdev):03d}STDEV" # "075STDEV"
            except KeyError: # key == 'classif_strategy' (new key)
                self.parsed_dict["(TrainConfig) dataset.classif_strategy"] = train_config["dataset"]["classif_strategy"]
            self.parsed_dict["(TrainConfig) dataset.param_name"] = train_config["dataset"]["param_name"]
            
            # model
            self.parsed_dict["(TrainConfig) model.name"] = train_config["model"]["model_name"]
            self.parsed_dict["(TrainConfig) model.pretrain_weights"] = train_config["model"]["pretrain_weights"]
            
            # train_opts
            self.parsed_dict["(TrainConfig) train_ratio"]  = train_config["train_opts"]["train_ratio"]
            self.parsed_dict["(TrainConfig) random_seed"]  = train_config["train_opts"]["random_seed"]
            self.parsed_dict["(TrainConfig) epochs"]       = train_config["train_opts"]["epochs"]
            self.parsed_dict["(TrainConfig) batch_size"]   = train_config["train_opts"]["batch_size"]
            
            # train_opts.optimizer
            self.parsed_dict["(TrainConfig) optimizer.lr"]           = train_config["train_opts"]["optimizer"]["learning_rate"]
            self.parsed_dict["(TrainConfig) optimizer.weight_decay"] = train_config["train_opts"]["optimizer"]["weight_decay"]
            
            # train_opts.lr_schedular
            self.parsed_dict["(TrainConfig) lr_schedular.enable"] = train_config["train_opts"]["lr_schedular"]["enable"]
            self.parsed_dict["(TrainConfig) lr_schedular.step"]   = train_config["train_opts"]["lr_schedular"]["step"]
            self.parsed_dict["(TrainConfig) lr_schedular.gamma"]  = train_config["train_opts"]["lr_schedular"]["gamma"]
            
            # train_opts.earlystop
            self.parsed_dict["(TrainConfig) earlystop.enable"]          = train_config["train_opts"]["earlystop"]["enable"]
            self.parsed_dict["(TrainConfig) earlystop.max_no_improved"] = train_config["train_opts"]["earlystop"]["max_no_improved"]
            
            # train_opts.data
            self.parsed_dict["(TrainConfig) data.use_hsv"]               = train_config["train_opts"]["data"]["use_hsv"]
            self.parsed_dict["(TrainConfig) data.aug_on_fly"]            = train_config["train_opts"]["data"]["aug_on_fly"]
            self.parsed_dict["(TrainConfig) data.forcing_balance"]       = train_config["train_opts"]["data"]["forcing_balance"]
            self.parsed_dict["(TrainConfig) data.forcing_sample_amount"] = train_config["train_opts"]["data"]["forcing_sample_amount"]
            
            # train_opts.debug_mode
            self.parsed_dict["(TrainConfig) debug_mode.enable"]      = train_config["train_opts"]["debug_mode"]["enable"]
            self.parsed_dict["(TrainConfig) debug_mode.rand_select"] = train_config["train_opts"]["debug_mode"]["rand_select"]

            # train_opts.cuda
            self.parsed_dict["(TrainConfig) cuda.index"]   = train_config["train_opts"]["cuda"]["index"]
            self.parsed_dict["(TrainConfig) cuda.use_amp"] = train_config["train_opts"]["cuda"]["use_amp"]
            
            # train_opts.cpu.multiworker
            self.parsed_dict["(TrainConfig) cpu.num_workers"] = train_config["train_opts"]["cpu"]["num_workers"]

        else:
            # dataset
            self.parsed_dict["(TrainConfig) dataset.name"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) dataset.palmskin_desc"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) dataset.gen_method"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) dataset.result_alias"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) dataset.classif_strategy"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) dataset.param_name"] = self.empty_cell_marker
            # model
            self.parsed_dict["(TrainConfig) model.name"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) model.pretrain_weights"] = self.empty_cell_marker
            # train_opts
            self.parsed_dict["(TrainConfig) train_ratio"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) random_seed"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) epochs"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) batch_size"] = self.empty_cell_marker
            # train_opts.optimizer
            self.parsed_dict["(TrainConfig) optimizer.lr"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) optimizer.weight_decay"] = self.empty_cell_marker
            # train_opts.lr_schedular
            self.parsed_dict["(TrainConfig) lr_schedular.enable"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) lr_schedular.step"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) lr_schedular.gamma"] = self.empty_cell_marker
            # train_opts.earlystop
            self.parsed_dict["(TrainConfig) earlystop.enable"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) earlystop.max_no_improved"] = self.empty_cell_marker
            # train_opts.data
            self.parsed_dict["(TrainConfig) data.use_hsv"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) data.aug_on_fly"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) data.forcing_balance"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) data.forcing_sample_amount"] = self.empty_cell_marker
            # train_opts.debug_mode
            self.parsed_dict["(TrainConfig) debug_mode.enable"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) debug_mode.rand_select"] = self.empty_cell_marker
            # train_opts.cuda
            self.parsed_dict["(TrainConfig) cuda.index"] = self.empty_cell_marker
            self.parsed_dict["(TrainConfig) cuda.use_amp"] = self.empty_cell_marker
            # train_opts.cpu.multiworker
            self.parsed_dict["(TrainConfig) cpu.num_workers"] = self.empty_cell_marker
        # -----------------------------------------------------------------------------------

    
    def parse_training_time(self): # For "(Training) Time" : "{ training time }_{ * sec }"
        key = "(Training) Time"
        target = self.scan_file(key)
        if target is not None:
            path, target_name = target
            with open(path, mode="r") as f_reader: training_time = float(f_reader.readline())
            self.parsed_dict[key] = training_time
        else:
            self.parsed_dict[key] = self.empty_cell_marker
        # -----------------------------------------------------------------------------------

    
    def parse_trainset_cnt(self): # For "(TrainSet)" : "{ dataset_* }_{ train_* }_{ valid_* }"
        key = "(TrainSet)"
        target = self.scan_file(key)
        if target is not None:
            path, target_name = target
            target_name_split = re.split("{|_|}| ", target_name)
            target_name_split = [text for text in target_name_split if text != ""] # rm "" in list
            self.parsed_dict[f"{key} Total"] = int(target_name_split[1])
            self.parsed_dict[f"{key} Train"] = int(target_name_split[3])
            self.parsed_dict[f"{key} Valid"] = int(target_name_split[5])
        else:
            self.parsed_dict[f"{key} Total"] = self.empty_cell_marker
            self.parsed_dict[f"{key} Train"] = self.empty_cell_marker
            self.parsed_dict[f"{key} Valid"] = self.empty_cell_marker
        # -----------------------------------------------------------------------------------

    
    def parse_testset_cnt(self): # For "(TestSet)"  : "{ datatest_* }_{ test_* }"
        key = "(TestSet)"
        target = self.scan_file(key)
        if target is not None:
            path, target_name = target
            target_name_split = re.split("{|_|}| ", target_name)
            target_name_split = [text for text in target_name_split if text != ""] # rm "" in list
            self.parsed_dict[f"{key} Test"] = int(target_name_split[1])
        else:
            self.parsed_dict[f"{key} Test"] = self.empty_cell_marker
        # -----------------------------------------------------------------------------------


    def parse_pred_f1_score(self, key:str): # e.g. {Logs}_PredByImg_maweavg_f1_*.toml, {Logs}_PredByFish_maweavg_f1_*.toml
        target = self.scan_file(f"({key}) Score")
        if target is not None:
            path, target_name = target
            with open(path, mode="r") as f_reader: score_dict = toml.load(f_reader)
            self.parsed_dict[f"({key}) L_f1"] = float(score_dict["L_f1"])
            self.parsed_dict[f"({key}) M_f1"] = float(score_dict["M_f1"])
            self.parsed_dict[f"({key}) S_f1"] = float(score_dict["S_f1"])
            self.parsed_dict[f"({key}) Micro_f1"]    = float(score_dict["micro_f1"])
            self.parsed_dict[f"({key}) Macro_f1"]    = float(score_dict["macro_f1"])
            self.parsed_dict[f"({key}) Weighted_f1"] = float(score_dict["weighted_f1"])
            self.parsed_dict[f"({key}) Maweavg_f1"]  = float(score_dict["maweavg_f1"])
        else:
            self.parsed_dict[f"({key}) L_f1"] = self.empty_cell_marker
            self.parsed_dict[f"({key}) M_f1"] = self.empty_cell_marker
            self.parsed_dict[f"({key}) S_f1"] = self.empty_cell_marker
            self.parsed_dict[f"({key}) Micro_f1"]    = self.empty_cell_marker
            self.parsed_dict[f"({key}) Macro_f1"]    = self.empty_cell_marker
            self.parsed_dict[f"({key}) Weighted_f1"] = self.empty_cell_marker
            self.parsed_dict[f"({key}) Maweavg_f1"]  = self.empty_cell_marker
        # -----------------------------------------------------------------------------------
    
    
    def parse_existing_target(self, key:str): # Just add `marker` to dict (No other advanced action)
        target = self.scan_file(key)
        if target is not None:
            path, target_name = target
            self.parsed_dict[key] = self.file_state_marker['found']
        else: 
            self.parsed_dict[key] = self.empty_cell_marker
        # -----------------------------------------------------------------------------------
    

    def parsed_dict2df(self):
        if self.parsed_df is None: self.parsed_df = pd.DataFrame(self.parsed_dict, index=[0])
        # -----------------------------------------------------------------------------------
    
    
    def parse(self) -> Union[pd.DataFrame, None]:
        
        self.log.info(f"{Fore.MAGENTA}Parsing{Style.RESET_ALL} '{Fore.GREEN}{self.prediction_dir_name}{Style.RESET_ALL}'")
        
        self.parsed_dict["(TrainConfig)"] = ""
        self.parse_train_config()
        # -----------------------------------------------------------------------------------
        self.parsed_dict["(TrainSet)"] = ""
        self.parse_trainset_cnt()
        self.parsed_dict["(TestSet)"] = ""
        self.parse_testset_cnt()
        # -----------------------------------------------------------------------------------
        self.parsed_dict["(Training)"] = ""
        self.parse_training_time()
        self.parse_existing_target("(Training) Train_Log")
        self.parse_existing_target("(Training) Valid_Log")
        self.parse_existing_target("(Training) Training_Log")
        self.parse_existing_target("(Training) Best_Valid_Log")
        self.parse_existing_target("(Training) Best_Model")
        self.parse_existing_target("(Training) Final_Model")
        self.parse_existing_target("(TrainFigure) Average_f1")
        self.parse_existing_target("(TrainFigure) Weighted_f1")
        self.parse_existing_target("(TrainFigure) Maweavg_f1")
        # -----------------------------------------------------------------------------------
        self.parsed_dict["(PredByImg)"] = ""
        self.parse_existing_target("(PredByImg) Report")
        self.parse_pred_f1_score("PredByImg")
        # -----------------------------------------------------------------------------------
        self.parsed_dict["(PredByFish)"] = ""
        self.parse_existing_target("(PredByFish) Report")
        self.parse_existing_target("(PredByFish) Predict_Ans")
        self.parse_pred_f1_score("PredByFish")
        # -----------------------------------------------------------------------------------
        self.parsed_dict["(Cam)"] = ""
        self.parse_existing_target("(Cam) Result_Dir")
        self.parse_existing_target("(Cam) Gallery_Dir")
        
        self.parsed_dict2df()
        
        if self.find_files_cnt > 1:
            self.log.info(f"{Fore.YELLOW}Done ( {self.find_files_cnt} targets are found )\n{Style.RESET_ALL}")
            return self.parsed_df
        else:
            raise ValueError(f"Can't not find any file in '{self.prediction_dir}'")