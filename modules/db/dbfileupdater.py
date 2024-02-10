import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from colorama import Back, Fore, Style
from rich import print

from ..shared.baseobject import BaseObject
from ..shared.config import load_config
from ..shared.utils import exclude_paths, exclude_tmp_paths
from .singlepredictionparser import SinglePredictionParser
from .utils import resolve_path_hyperlink
# -----------------------------------------------------------------------------/


class DBFileUpdater(BaseObject):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("DB Excel Updater")
        self._single_pred_parser = SinglePredictionParser(display_on_CLI)
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self._config = load_config("6.update_db_excel.toml")
        self._col_version = self._config["column_version"]
        self._state_mark: dict[str, str] = self._config["state_mark"]
        self._possible_item_dict: dict[str, str] = \
                                        self._config["possible_item"]
        
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _reset_attrs(self):
        """ Reset below attributes
            >>> self.dirs_state: dict[str, list[Union[Path, tuple[str, pd.DataFrame], str]]]
            >>> self.model_prediction: Path
            >>> self._current_db: pd.DataFrame
            >>> self._csv_path: Path
            >>> self._excel_path: Path
        """
        self.dirs_state: dict[str, list[Union[Path, tuple[str, pd.DataFrame], str]]] = {}
        self.dirs_state["Unpredicted History Dir"] = [] # list[path]
        self.dirs_state["Updated Prediction Dir"] = [] # list[(index, diff_df)]
        self.dirs_state["New Prediction Dir"] = [] # list[index]
        self.dirs_state["Deleted Prediction Dir"] = [] # list[index]
        # ↑ above `dirs` objects are printed at the end of process
        
        self.model_prediction: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("model_prediction")
        
        # self._current_db
        ver_test_dir = \
            list(self.model_prediction.glob(f"{self._col_version}/.ver_test/*"))[-1]
        print(f"Latest Version Test Dir: '{ver_test_dir}'")
         # ↑ always build column with latest (final) version test directory
        self._current_db: pd.DataFrame = \
            self._single_pred_parser.parse(ver_test_dir).set_index("Prediction_ID")
        self._current_db.index.values[0] = 'std_col' # reset index for easily delete later
        
        db_root: Path = Path(self._path_navigator.dbpp.dbpp_config["root"])
        self._csv_path: Path = db_root.joinpath(r"{DB}_Predictions.csv")
        self._excel_path: Path = db_root.joinpath(r"{DB}_Predictions.xlsx")
        # ---------------------------------------------------------------------/


    def update(self):
        """
        """
        self._cli_out.divide()
        
        self._reset_attrs()
        self._get_prediction_dirs_dict()
        self._bulid_current_db()
        self._set_previous_db()
        
        if self._previous_db is not None:
            self._detect_changed()
        
        self._display_changed_on_CLI()
        
        # save file
        self._current_db.to_csv(self._csv_path, encoding='utf_8_sig')
        self._current_db.to_excel(self._excel_path, engine="openpyxl")
        
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _get_prediction_dirs_dict(self) -> dict[str, Path]:
        """
        """
        # scan dir
        found_list: list[Path] = \
            sorted(self.model_prediction.glob("**/{ training time }_{ * sec }"),
                   key=lambda x: x.parts[-2])
        found_list = exclude_tmp_paths(found_list)
        found_list = exclude_paths(found_list, [".ver_test"])
        
        # create dict
        prediction_dirs_dict: dict[str, Path] = {}
        for file in found_list:
            
            # >>> prediction dir <<<
            prediction_dir = file.parent
            
            # >>> key: Prediction_ID <<<
            name_split = re.split("{|}", prediction_dir.parts[-1])
            if "Test" in name_split[1]:
                """
                """
                time_stamp = name_split[0] # e.g.: '20240203_22_12_56_'
                final_epoch = name_split[3].split("_")[0] # e.g.: '69_epochs_AugOnFly'
                model_state = name_split[5] # e.g.: 'best'
                
                pred_id = \
                    (f"{time_stamp} | {model_state:5} | {final_epoch}_epoch")
                
                # check if duplicated
                if pred_id in prediction_dirs_dict:
                    raise ValueError(f"{Fore.RED}{Back.BLACK} Detect duplicated 'Prediction_ID': '{pred_id}', "
                                     f"'Prediction_ID' should be unique in `{self.model_prediction.parts[-1]}`\n"
                                     f"Dir_1: '{prediction_dirs_dict[pred_id]}'\n"
                                     f"Dir_2: '{prediction_dir}'")
                # check_ok, add to dict
                prediction_dirs_dict[pred_id] = prediction_dir
                
            else:
                # Unpredicted History Dir
                self.dirs_state["Unpredicted History Dir"].append(prediction_dir)
        
        # set dict as attr
        self._prediction_dirs_dict = prediction_dirs_dict
        # ---------------------------------------------------------------------/


    def _bulid_current_db(self):
        """
        """
        for k, v in self._prediction_dirs_dict.items():
            """ parse
            """
            new_row = \
                self._single_pred_parser.parse(v).set_index("Prediction_ID")
            
            if self._current_db is None:
                """ init
                """
                self._current_db = deepcopy(new_row)
            else:
                """ concat
                """
                self._current_db = pd.concat([self._current_db, new_row])
        
        self._current_db.drop("std_col", inplace=True) # 刪除為了指定 column 順序而引入的 row
        self._current_db.sort_index(inplace=True)
        # ---------------------------------------------------------------------/


    def _set_previous_db(self):
        """
        """
        if self._csv_path.exists():
            
            self._previous_db: pd.DataFrame = \
                pd.read_csv(self._csv_path, encoding='utf_8_sig',
                            index_col="Prediction_ID")
        else:
            self._previous_db = None
        # ---------------------------------------------------------------------/


    def _detect_changed(self):
        """
        """
        current_db = self._current_db.copy().fillna(self._state_mark["empty_cell"])
        current_db = current_db.astype(str) # 轉成 str 減少 float 容易不相等的問題
        previous_db = self._previous_db.copy().fillna(self._state_mark["empty_cell"])
        previous_db = previous_db.astype(str) # 轉成 str 減少 float 容易不相等的問題
        
        for index in current_db.index:
            
            if index in previous_db.index:
                # unchanged / updated
                row: pd.Series = current_db.loc[index]
                previous_row: pd.Series = previous_db.loc[index]
                diff_df = row.compare(previous_row)
                
                if diff_df.empty:
                    """ unchanged """
                    pass
                else:
                    """ updated """
                    self.dirs_state["Updated Prediction Dir"].append((index, diff_df))
                
                previous_db.drop(index, inplace=True)
            else:
                # new row
                self.dirs_state["New Prediction Dir"].append(index)
        
        # remain in `previous_db`
        self.dirs_state["Deleted Prediction Dir"].extend(list(previous_db.index))
        # ---------------------------------------------------------------------/


    def _display_changed_on_CLI(self):
        """
        """
        changed_cnt = 0
        
        # >>> Updated <<<
        if len(self.dirs_state["Updated Prediction Dir"]) > 0:
            changed_cnt += 1
            # display once
            self._cli_out.divide()
            print("[yellow]*** Updated Prediction ***\n")
            
            for index, diff_df in self.dirs_state["Updated Prediction Dir"]:
                
                link = self._previous_db.loc[index, "Local_Path"]
                _, path = resolve_path_hyperlink(link)
                
                new_link = self._current_db.loc[index, "Local_Path"]
                _, new_path = resolve_path_hyperlink(new_link)
                
                path_state_msg = self._get_path_state_msg(diff_df)
                file_state_msg, \
                    value_state_msg = self._get_item_state_msg(diff_df)
                
                # CLI Output
                print(f"{path_state_msg}[#2596be]'{path}'")
                if file_state_msg: print(file_state_msg)
                if value_state_msg: print(value_state_msg)
                print(f"--> [#be4d25]'{new_path}'\n")
        
        
        # >>> New <<<
        if len(self.dirs_state["New Prediction Dir"]) > 0:
            changed_cnt += 1
            # display once
            self._cli_out.divide()
            print("[yellow]*** New Prediction ***\n")
            
            for index in self.dirs_state["New Prediction Dir"]:
                new_link = self._current_db.loc[index, "Local_Path"]
                print(f"Dir: '{resolve_path_hyperlink(new_link)[1]}'")
        
        
        # >>> Deleted <<<
        if len(self.dirs_state["Deleted Prediction Dir"]) > 0:
            changed_cnt += 1
            # display once
            self._cli_out.divide()
            print("[yellow]*** Deleted Prediction ***\n")
            
            for index in self.dirs_state["Deleted Prediction Dir"]:
                link = self._previous_db.loc[index, "Local_Path"]
                print(f"Dir: [magenta]'{resolve_path_hyperlink(link)[1]}'")
        
        
        if (self._previous_db is not None) and (changed_cnt == 0):
            self._cli_out.divide()
            self._cli_out.write(f"{Fore.YELLOW} --- No Changed --- {Style.RESET_ALL}")
        
        
        # >>> Unpredicted <<<
        if len(self.dirs_state["Unpredicted History Dir"]) > 0:
            # display once
            self._cli_out.divide()
            print("[yellow]*** Unpredicted History ***\n")
            
            for path in self.dirs_state["Unpredicted History Dir"]:
                print(f"Dir: [orange1]'{path}'")
        # ---------------------------------------------------------------------/


    def _get_path_state_msg(self, diff_df:pd.DataFrame):
        """
        """
        state_msg = "Dir: "
        
        if "Local_Path" in diff_df.index:
            # path is different
            cur_link = diff_df.loc["Local_Path", "self"]
            pre_link = diff_df.loc["Local_Path", "other"]
            
            _, cur_path = resolve_path_hyperlink(cur_link)
            _, pre_path = resolve_path_hyperlink(pre_link)
            
            if cur_path.parent == pre_path.parent:
                state_msg = "Rename Dir, "
            else:
                state_msg = "Move Dir, "
            
            diff_df.drop(["Local_Path"], inplace=True) # 已經判斷完，可刪除
        
        return state_msg
        # ---------------------------------------------------------------------/


    def _get_item_state_msg(self, diff_df:pd.DataFrame) -> tuple[str, str]:
        """
        """
        if "Files" in diff_df.index:
            diff_df.drop("Files", inplace=True)
        
        rm_file = []
        add_file = []
        
        rm_value = []
        add_value = []
        changed_value = []
        
        for index in diff_df.index:
            
            if index in self._possible_item_dict:
                # column represent a file / dir
                if (diff_df.loc[index, "self"] == self._state_mark["empty_cell"]) and \
                    (diff_df.loc[index, "other"] != self._state_mark["empty_cell"]):
                    rm_file.append(index)
                #
                elif (diff_df.loc[index, "self"] != self._state_mark["empty_cell"]) and \
                    (diff_df.loc[index, "other"] == self._state_mark["empty_cell"]):
                    add_file.append(index)
            else:
                # column: value
                if (diff_df.loc[index, "self"] == self._state_mark["empty_cell"]) and \
                    (diff_df.loc[index, "other"] != self._state_mark["empty_cell"]):
                    rm_value.append(index)
                #
                elif (diff_df.loc[index, "self"] != self._state_mark["empty_cell"]) and \
                    (diff_df.loc[index, "other"] == self._state_mark["empty_cell"]):
                    add_value.append(index)
                #
                else:
                    changed_value.append(index)
        
        
        file_state_msg = []
        if len(add_file) > 0:
            file_state_msg.append(f"+ Add {len(add_file)} item: {add_file}")
        if len(rm_file) > 0:
            file_state_msg.append(f"- Remove {len(rm_file)} item: {rm_file}")
        file_state_msg = "\n".join(file_state_msg)
        
        
        value_state_msg = []
        if len(add_value) > 0:
            value_state_msg.append(f"+ add {len(add_value)} value, {add_value}")
        if len(rm_value) > 0:
            value_state_msg.append(f"- remove {len(rm_value)} value, {rm_value}")
        if len(changed_value) > 0:
            value_state_msg.append(f"· changed {len(changed_value)} value, {changed_value}")
        value_state_msg = "\n".join(value_state_msg)
        
        
        return file_state_msg, value_state_msg
        # ---------------------------------------------------------------------/