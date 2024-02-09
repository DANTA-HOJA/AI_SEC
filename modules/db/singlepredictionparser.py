import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from colorama import Fore, Style

from ..assert_fn import assert_is_pathobj
from ..db.utils import create_path_hyperlink, flatten_dict
from ..shared.baseobject import BaseObject
from ..shared.config import load_config
# -----------------------------------------------------------------------------/

#  INFO: 
# `train_config` parser `stdev` -> `classif_strategy`, e.g. 0.75_STDEV -> 075STDEV
#   --> dataset_config 只有更改為 "只需要 xlsx_file"，cropping 的動作都沒有修改

# WARNING: `dataset_config` 多一個 key `sheet_name` ( 不是很重要了， `dataset_config` 的資訊都可以透過 `train_config` 重組)

# WARNING: `history_dir`: `{Logs}_train.xlsx` + `{Logs}_valid.xlsx` = `{Logs}_training_log.xlsx`

# WARNING: `training_trend` 有三種 : 1. `training_trend_average_f1.png`
#                                         2. `training_trend_weighted_f1.png` 
#                                         3. `training_trend_maweavg_f1.png`

# WARNING: 舊的 train_config 找不到 `result_alias` 的話自動將 value 設為 "( RGB_HE_fusion )"


#  TODO:  透過 history name 判斷哪些還沒更新過，是 `Update` 不是 Rewrite（保存可能使用 PKL or XLSX）

#  TODO:  data operate 時的 parameters 要 parse 進來

#  TODO: 
# 將 dataset 和 data 的資訊 parser 進來，檢查 column 是否重複，若有重複的 column 要刪除，
# 例如："(TrainSet)" : "{ dataset_* }_{ train_* }_{ valid_* }" 在 dataset info 加進來之後應該要刪除

#  TODO: 
# 產生專門 render 路徑的 script（HYPERLINK text 的 funcion）
# data, dataset 的路徑都可以由 train_config 重新組合，
# 查巡當下的電腦裡面是否有對應的 data, dataset 存在，有對應的檔案即可替原始的 content 加上 HYPERLINK
# 每台 PC 上都能重新 render 路徑

# -----------------------------------------------------------------------------/


class SinglePredictionParser(BaseObject):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Single Prediction Parser")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self._config = load_config("6.update_db_excel.toml")
        self._state_mark: dict[str, str] = self._config["state_mark"]
        self._possible_item_dict: dict[str, str] = \
                                        self._config["possible_item"]
        
        # ---------------------------------------------------------------------
        # """ actions """
        
        max_length = 0
        for key in self._possible_item_dict.keys():
            if len(key) > max_length:
                max_length = len(key)
        self._key_max_length = max_length+1
        # ---------------------------------------------------------------------/


    def _reset_attrs(self):
        """ Reset below attributes
            >>> self._found_files_cnt
            >>> self._parsed_dict
            >>> self._alt_name_dict
        """
        self._found_files_cnt: int = 0
        self._parsed_dict: dict = {}
        self._alt_name_dict: dict[str, list[str]] = \
                                        deepcopy(self._config["alt_name"])
        # ---------------------------------------------------------------------/


    def parse(self, prediction_dir:Path) -> Union[pd.DataFrame, None]:
        """
        """
        self._reset_attrs()
        self._handle_prediction_dir(prediction_dir)
        
        self._item_path_dict = self._scan_items()
        self._handle_time()
        self._handle_trainingdata_cnt()
        self._handle_testdata_cnt()
        self._handle_training_config()
        self._handle_score("TestByImg")
        self._handle_score("TestByFish")
        
        self._parsed_dict["Files"] = self._found_files_cnt
        if self._found_files_cnt > 1:
            self._cli_out.write(f"{Fore.YELLOW}Done "
                                f"( {self._found_files_cnt} targets are found )\n"
                                f"{Style.RESET_ALL}")
            return pd.DataFrame(self._parsed_dict, index=[0])
        else:
            self._cli_out.write(f"Can't find any file in '{self._prediction_dir}'")
            return None
        # ---------------------------------------------------------------------/


    def _handle_prediction_dir(self, prediction_dir:Path):
        """
        """
        try:
            assert_is_pathobj(prediction_dir)
            self._prediction_dir = prediction_dir
        except TypeError as e:
            raise TypeError(str(e).replace("The given path", "`prediction_dir`"))
        
        if not self._prediction_dir.exists():
            raise FileNotFoundError(f"Can't reach the folder '{self._prediction_dir}'")
        
        self._cli_out.write(f"{Fore.MAGENTA}Parsing{Style.RESET_ALL} "
                            f"'{Fore.GREEN}{self._prediction_dir}'"
                            f"{Style.RESET_ALL}")
        
        name_split = re.split("{|}", prediction_dir.parts[-1])
        time_stamp = name_split[0] # e.g.: '20240203_22_12_56_'
        final_epoch = name_split[3].split("_")[0] # e.g.: '69_epochs_AugOnFly'
        model_state = name_split[5] # e.g.: 'best'
        
        # col: Prediction_ID
        self._parsed_dict["Prediction_ID"] = \
                (f"{time_stamp} | {model_state:5} | {final_epoch}_epoch")
        
        # col: Version
        model_prediction: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("model_prediction")
        rel_path = prediction_dir.relative_to(model_prediction).parts[0]
        self._parsed_dict["Version"] = rel_path.split("_")[0]
        
        # col: state
        self._parsed_dict["State"] = model_state
        
        # col: (HyperLink) Local_Path
        self._parsed_dict["Local_Path"] = \
                        create_path_hyperlink("local_path", prediction_dir)
        
        # col: TrainingConfig.Note (move column forward)
        self._parsed_dict["Files"] = ""
        self._parsed_dict["TrainingConfig.Note"] = ""
        self._parsed_dict["TestByImg.Maweavg_f1"] = 0.0
        self._parsed_dict["TestByFish.Maweavg_f1"] = 0.0
        # ---------------------------------------------------------------------/


    def _get_item_path(self, item:tuple[str, str]):
        """

        Args:
            item (str): `(key, value)` in self.possible_file_dict

        Raises:
            ValueError: multiple path are found.

        Returns:
            Path: _description_
        """
        key, name = item
        found_list = list(self._prediction_dir.glob(name))
        
        if len(found_list) > 1:
            raise ValueError(f"'{name}' should be a unique item in each prediction folder, "
                             f"but found {len(found_list)} same items")
        elif len(found_list) == 1:
            return found_list[0]
        else:
            try:
                alt_name = self._alt_name_dict[key].pop(0)
                return self._get_item_path((key, alt_name))
            except (IndexError, KeyError):
                return None
        # ---------------------------------------------------------------------/


    def _scan_items(self):
        """
        """
        path_dict: dict[str, Union[Path, None]] = deepcopy(self._possible_item_dict)
        
        for k, v in self._possible_item_dict.items():
            path = self._get_item_path((k, v))
            if path is None:
                path_dict[k] = None
                self._parsed_dict[k] = self._state_mark["empty_cell"]
                self._cli_out.write(f"{Fore.BLACK}{k:{self._key_max_length}}: "
                                    f"[ {self._state_mark['not_found']} ] "
                                    f"{self._state_mark['empty_cell']}"
                                    f"{Style.RESET_ALL}")
            else:
                self._found_files_cnt += 1
                path_dict[k] = path
                self._parsed_dict[k] = self._state_mark["found"]
                self._cli_out.write(f"{k:{self._key_max_length}}: "
                                    f"[ {self._state_mark['found']} ] "
                                    f"{path.parts[-1]}")
        
        return path_dict
        # ---------------------------------------------------------------------/


    def _handle_trainingdata_cnt(self):
        """
        """
        key = "(INFO) TrainingData"
        path = self._item_path_dict[key]
        
        if path:
            target_name_split = re.split("{|_|}| ", path.parts[-1])
            target_name_split = [text for text in target_name_split if text != ""] # rm "" in list
            self._parsed_dict[f"TrainingData.Total"] = int(target_name_split[1])
            self._parsed_dict[f"TrainingData.Train"] = int(target_name_split[3])
            self._parsed_dict[f"TrainingData.Valid"] = int(target_name_split[5])
        else:
            self._parsed_dict[f"TrainingData.Total"] = self._state_mark['empty_cell']
            self._parsed_dict[f"TrainingData.Train"] = self._state_mark['empty_cell']
            self._parsed_dict[f"TrainingData.Valid"] = self._state_mark['empty_cell']
        # ---------------------------------------------------------------------/


    def _handle_testdata_cnt(self):
        """
        """
        key = "(INFO) TestData"
        path = self._item_path_dict[key]
        
        if path:
            target_name_split = re.split("{|_|}| ", path.parts[-1])
            target_name_split = [text for text in target_name_split if text != ""] # rm "" in list
            self._parsed_dict[f"TestData.Test"] = int(target_name_split[1])
        else:
            self._parsed_dict[f"TestData.Test"] = self._state_mark['empty_cell']
        # ---------------------------------------------------------------------/


    def _handle_time(self):
        """
        """
        key = "(INFO) Time"
        path = self._item_path_dict[key]
        
        if path:
            with open(path, mode="r") as f_reader:
                self._parsed_dict["Training.Time"] = \
                            round(float(f_reader.readline()), 5)
        else:
            self._parsed_dict["Training.Time"] = self._state_mark['empty_cell']
        # ---------------------------------------------------------------------/


    def _handle_training_config(self):
        """
        """
        key = "(TOML) Config"
        path = self._item_path_dict[key]
        
        if path:
            config = load_config(path)
            new_config = flatten_dict(config, "TrainingConfig")
        
        for k, v in new_config.items():
            self._parsed_dict[k] = v
        
        # TODO:  key 在不同版本下，需額外處理
        # ---------------------------------------------------------------------/


    def _handle_score(self, key:str):
        """
        """
        path = self._item_path_dict[f"(SCORE) {key}"]
        
        if path:
            score_dict = load_config(path)
            self._parsed_dict[f"{key}.L_f1"] = float(score_dict["L_f1"])
            self._parsed_dict[f"{key}.M_f1"] = float(score_dict["M_f1"])
            self._parsed_dict[f"{key}.S_f1"] = float(score_dict["S_f1"])
            self._parsed_dict[f"{key}.Micro_f1"]    = float(score_dict["micro_f1"])
            self._parsed_dict[f"{key}.Macro_f1"]    = float(score_dict["macro_f1"])
            self._parsed_dict[f"{key}.Weighted_f1"] = float(score_dict["weighted_f1"])
            self._parsed_dict[f"{key}.Maweavg_f1"]  = float(score_dict["maweavg_f1"])
        else:
            self._parsed_dict[f"{key}.L_f1"] = self._state_mark['empty_cell']
            self._parsed_dict[f"{key}.M_f1"] = self._state_mark['empty_cell']
            self._parsed_dict[f"{key}.S_f1"] = self._state_mark['empty_cell']
            self._parsed_dict[f"{key}.Micro_f1"]    = self._state_mark['empty_cell']
            self._parsed_dict[f"{key}.Macro_f1"]    = self._state_mark['empty_cell']
            self._parsed_dict[f"{key}.Weighted_f1"] = self._state_mark['empty_cell']
            self._parsed_dict[f"{key}.Maweavg_f1"]  = self._state_mark['empty_cell']
        # ---------------------------------------------------------------------/