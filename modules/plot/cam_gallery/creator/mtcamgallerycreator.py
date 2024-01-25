import os
import re
import sys
import threading
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple, Union

from colorama import Fore, Style
from tqdm.auto import tqdm

from .camgallerycreator import CamGalleryCreator
# -----------------------------------------------------------------------------/


class MtCamGalleryCreator(CamGalleryCreator):

    def __init__(self, max_str_len_dict:Dict[str, int], lock:Lock,
                 progressbar:tqdm, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(CamGalleryCreator, self).__init__(display_on_CLI)
        self._cli_out._set_logger("Mt Cam Gallery Creator")
        
        self._lock: Lock = lock
        self._progressbar: tqdm = progressbar
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.max_str_len_dict: Dict[str, int] = max_str_len_dict
        
        # ---------------------------------------------------------------------
        # """ actions """
        
        if len(threading.current_thread().name) > self.max_str_len_dict["thread_name"]:
            self.max_str_len_dict["thread_name"] = \
                                        len(threading.current_thread().name)
        # ---------------------------------------------------------------------/


    def mt_run(self, sub_fish_dsnames:List[str], config:Union[str, Path]):
        """

        Args:
            sub_fish_dsname_list (List[str]):
                a subset of all test dsname, assign by `MtCamGalleryExecutor`
            config (Union[str, Path]): a toml file.
        """
        with self._lock: # initial 時會有很多東西動到 File I/O，所以要上鎖
            self._set_attrs(config)
            self._create_rank_dirs()
            self._cli_out.write(f"{Fore.MAGENTA}{threading.current_thread().name} "
                                f"{Fore.BLUE}{threading.current_thread().native_id} "
                                f"{Fore.GREEN}initial{Style.RESET_ALL} completed")
            self._cli_out.write(f"{self}\n")
        
        
        for fish_dsname in sub_fish_dsnames:
            self._progressbar.desc = (f"Generate {Fore.YELLOW}'{fish_dsname:{self.max_str_len_dict['fish_dsname']}}'{Style.RESET_ALL} "
                                      f"( {Fore.MAGENTA}{threading.current_thread().name:{self.max_str_len_dict['thread_name']}}{Style.RESET_ALL} ) ")
            self._progressbar.refresh()
            self.gen_single_cam_gallery(fish_dsname)
        # ---------------------------------------------------------------------/


    def _set_cam_gallery_dir(self):
        """ 移除資料夾檢查，改從 `MtCamGalleryExecutor` 檢查
        """
        self.cam_gallery_dir = self.history_dir.joinpath("+---CAM_Gallery")
        # ---------------------------------------------------------------------/