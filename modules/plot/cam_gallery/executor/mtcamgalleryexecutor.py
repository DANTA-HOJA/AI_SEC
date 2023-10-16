import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
import traceback
from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from tomlkit.toml_document import TOMLDocument
from colorama import Fore, Back, Style
from tqdm.auto import tqdm

from .utils import divide_fish_dsname_in_group
from ..creator.mtcamgallerycreator import MtCamGalleryCreator
from ....data.dataset import dsname
from ....shared.clioutput import CLIOutput
from ....shared.pathnavigator import PathNavigator
from ....shared.config import load_config
from ....assert_fn import assert_0_or_1_history_dir
# -----------------------------------------------------------------------------/


class MtCamGalleryExecutor:


    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        self._path_navigator = PathNavigator()
        self._cli_out = CLIOutput(display_on_CLI, 
                                  logger_name="Mt Cam Gallery Executor")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------/



    def _set_attrs(self, config_file:Union[str, Path]):
        """
        """
        self.config: Union[dict, TOMLDocument] = load_config(config_file, cli_out=self._cli_out)
        self._set_config_attrs()
        self._set_history_dir()
        self._set_cam_result_root()
        self._set_cam_gallery_dir()
        
        self.max_str_len_dict: Dict[str, int] = {"fish_dsname": 0,
                                                 "thread_name": 0}
        # ---------------------------------------------------------------------/



    def run(self, config_file:Union[str, Path]="4.make_cam_gallery.toml"):
        """
        """
        self._cli_out.divide()
        self._set_attrs(config_file)
        
        fish_dsname_list_group = self.divide_fish_dsnames()
        
        self._cli_out.divide()
        progressbars = \
            [ tqdm(total=len(fish_dsname_list), desc=f"[ {self._cli_out.logger_name} ] ") \
                for fish_dsname_list in fish_dsname_list_group ]
        
        lock = Lock()
        with ThreadPoolExecutor(max_workers=self.worker) as t_pool:
            futures = [ t_pool.submit(
                            self.mt_task, *(fish_dsname_list,
                                            self.max_str_len_dict,
                                            lock, progressbars[i],
                                            self._cli_out._display_on_CLI)
                        ) for i, fish_dsname_list in enumerate(fish_dsname_list_group) ]
            
            try:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            
            except Exception:
                for progressbar in progressbars: progressbar.close()
                self._cli_out.new_line()
                self._cli_out.write(f"{traceback.format_exc()}") # 輸出異常訊息

            else:
                for progressbar in progressbars: progressbar.close()
                self._cli_out.new_line()
                self._cli_out.write(f"Done: '{self.cam_gallery_dir}'")
                self._cli_out.new_line()
        # ---------------------------------------------------------------------/



    def _set_config_attrs(self):
        """
        """
        """ [model_prediction] """
        self.model_time_stamp: str = self.config["model_prediction"]["time_stamp"]
        self.model_state: str = self.config["model_prediction"]["state"]
        
        """ [multiprocessing] """
        self.worker = self.config["multiprocessing"]["worker"]
        # ---------------------------------------------------------------------/



    def _set_history_dir(self):
        """
        """
        if self.model_state not in ["best", "final"]:
            raise ValueError(f"config: `model_prediction.state`: "
                             f"'{self.model_state}', accept 'best' or 'final' only\n")
        
        model_prediction: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("model_prediction")
        
        best_found = []
        final_found = []
        found_list = list(model_prediction.glob(f"{self.model_time_stamp}*"))
        for i, path in enumerate(found_list):
            if f"{{best}}" in str(path): best_found.append(found_list.pop(i))
            if f"{{final}}" in str(path): final_found.append(found_list.pop(i))

        if self.model_state == "best" and best_found:
            assert_0_or_1_history_dir(best_found, self.model_time_stamp, self.model_state)
            self.history_dir = best_found[0]
            return
        
        if self.model_state == "final" and final_found:
            assert_0_or_1_history_dir(final_found, self.model_time_stamp, self.model_state)
            self.history_dir = final_found[0]
            return
        
        assert_0_or_1_history_dir(found_list, self.model_time_stamp, self.model_state)
        if found_list:
            self.history_dir = found_list[0]
            return
        else:
            raise ValueError("No `history_dir` matches the provided config")
        # ---------------------------------------------------------------------/



    def _set_cam_result_root(self):
        """
        """
        self.cam_result_root = self.history_dir.joinpath("cam_result")
        if not self.cam_result_root.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find directory: 'cam_result/' "
                                    f"run `3.2.{{TestByFish}}_vit_b_16.py` and set `cam.enable` = True. "
                                    f"{Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/



    def _set_cam_gallery_dir(self):
        """
        """
        self.cam_gallery_dir = self.history_dir.joinpath("!--- CAM Gallery")
        if self.cam_gallery_dir.exists():
            raise FileExistsError(f"{Fore.RED}{Back.BLACK} Directory already exists: '{self.cam_gallery_dir}'. "
                                  f"To re-generate, please delete it manually. "
                                  f"{Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/



    def divide_fish_dsnames(self) -> List[List[str]]:
        """
        """
        fish_dsname_list = [ str(path).split(os.sep)[-1] for path in list(self.cam_result_root.glob("*")) ]
        fish_dsname_list = sorted(fish_dsname_list, key=dsname.get_dsname_sortinfo)
        # fish_dsname_list = fish_dsname_list[:32] # for debug
        
        # update `self.max_str_len_dict["fish_dsname"]`
        for fish_dsname in fish_dsname_list:
            if len(fish_dsname) > self.max_str_len_dict["fish_dsname"]:
                self.max_str_len_dict["fish_dsname"] = len(fish_dsname)
        
        fish_dsname_list_group = \
            divide_fish_dsname_in_group(fish_dsname_list, self.worker)
            
        return fish_dsname_list_group
        # ---------------------------------------------------------------------/



    def mt_task(self, fish_dsname_list, max_str_len_dict,
                lock, progressbar, display_on_CLI):
        """
        """
        mt_cam_gallery_creator = MtCamGalleryCreator(max_str_len_dict,
                                                     lock, progressbar,
                                                     display_on_CLI)
        mt_cam_gallery_creator.mt_run(fish_dsname_list)
        # ---------------------------------------------------------------------/