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
from ..shared.utils import exclude_tmp_paths
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
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _reset_attrs(self):
        """ Reset below attributes
            >>> self.unpredicted_history_dirs: list[Path]
            >>> self.dirs_state: dict[str, list[Path]]
            >>> self._current_db: Union[None, pd.DataFrame]
            >>> self._csv_path: Path
            >>> self._excel_path: Path
        """
        self.unpredicted_history_dirs: list[Path] = []
        self.dirs_state: dict[str, list[Path]] = {}
        self.dirs_state["New Prediction Dir"] = []
        self.dirs_state["Updated Prediction Dir"] = []
        self.dirs_state["Deleted Prediction Dir"] = []
        # ↑ above `dirs` objects are printed at the end of process
        
        self._current_db: Union[None, pd.DataFrame] = None
        
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
        model_prediction: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("model_prediction")
        
        # scan dir
        found_list: list[Path] = \
            sorted(model_prediction.glob("**/{ training time }_{ * sec }"),
                   key=lambda x: x.parts[-2])
        found_list = exclude_tmp_paths(found_list)
        found_list.insert(0, found_list.pop(-1))
        # ↑ Always use lastest history to bulid `DataFrame` column.
        
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
                                     f"'Prediction_ID' should be unique in `{model_prediction.parts[-1]}`\n"
                                     f"Dir_1: '{prediction_dirs_dict[pred_id]}'\n"
                                     f"Dir_2: '{prediction_dir}'")
                # check_ok, add to dict
                prediction_dirs_dict[pred_id] = prediction_dir
                
            else:
                # Unpredicted History Dir
                self.unpredicted_history_dirs.append(prediction_dir)
        
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
        current_db = self._current_db.copy().astype(str)
        previous_db = self._previous_db.copy().astype(str)
        
        for index in current_db.index:
            
            if index in previous_db.index:
                # unchanged / updated
                row: pd.Series = current_db.loc[index]
                previous_row: pd.Series = previous_db.loc[index]
                
                if row.equals(previous_row):
                    """ unchanged """
                    pass
                else:
                    """ updated """
                    self.dirs_state["Updated Prediction Dir"].append(index)
                
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
        
        # >>> Update <<<
        if len(self.dirs_state["Updated Prediction Dir"]) > 0:
            changed_cnt += 1
            # display once
            self._cli_out.divide()
            print("[yellow]*** Updated Prediction ***\n")
            
            for index in self.dirs_state["Updated Prediction Dir"]:
                
                link = self._previous_db.loc[index, "Local_Path"]
                _, path = resolve_path_hyperlink(link)
                path_cnt = int(self._previous_db.loc[index, "Files"])
                
                new_link = self._current_db.loc[index, "Local_Path"]
                _, new_path = resolve_path_hyperlink(new_link)
                new_path_cnt = int(self._current_db.loc[index, "Files"])
                
                if new_path_cnt == path_cnt: update_flag = "Rename Only"
                elif new_path_cnt > path_cnt: update_flag = f"Upgrade"
                else: update_flag = "Downgrade"
                
                print(f"{update_flag}: [#2596be]'{path}'\n [#FFFFFF]--> [#be4d25]'{new_path}'\n")
        
        
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
        
        
        if changed_cnt == 0:
            self._cli_out.divide()
            self._cli_out.write(f"{Fore.YELLOW} --- No Changed --- {Style.RESET_ALL}")
        
        
        # >>> Unpredicted <<<
        if len(self.unpredicted_history_dirs) > 0:
            # display once
            self._cli_out.divide()
            print("[yellow]*** Unpredicted History ***\n")
            
            for path in self.unpredicted_history_dirs:
                print(f"Dir: [orange1]'{path}'")
        # ---------------------------------------------------------------------/