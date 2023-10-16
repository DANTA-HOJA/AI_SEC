import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

from colorama import Fore, Style
from tqdm.auto import tqdm
import threading
from threading import Lock

from .camgallerycreator import CamGalleryCreator
from ....shared.clioutput import CLIOutput
from ....shared.pathnavigator import PathNavigator
# -----------------------------------------------------------------------------/


class MtCamGalleryCreator(CamGalleryCreator):


    def __init__(self, max_str_len_dict:int, lock:Lock,
                 progressbar:tqdm, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        self.lock: Lock = lock
        self.progressbar: tqdm = progressbar
        
        with self.lock:
            self._path_navigator = PathNavigator()
        
        self._cli_out = CLIOutput(display_on_CLI, 
                                  logger_name="Mt Cam Gallery Creator")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.max_str_len_dict: Dict[str, int] = max_str_len_dict
        with self.lock:
            if len(threading.current_thread().name) > self.max_str_len_dict["thread_name"]:
                self.max_str_len_dict["thread_name"] = len(threading.current_thread().name)
        # ---------------------------------------------------------------------/



    def mt_run(self, sub_fish_dsname_list:List[str],
               config_file:Union[str, Path]="4.make_cam_gallery.toml"):
        """
        """
        with self.lock:
            self._set_attrs(config_file)
            self.create_rank_dirs()
            self._cli_out.write(f"{Fore.MAGENTA}{threading.current_thread().name} "
                                f"{Fore.BLUE}{threading.current_thread().native_id} "
                                f"{Fore.GREEN}initial{Style.RESET_ALL} completed")
            self._cli_out.write(f"{self}\n")
        
        
        for fish_dsname in sub_fish_dsname_list:
            self.progressbar.desc = (f"Generate {Fore.YELLOW}'{fish_dsname:{self.max_str_len_dict['fish_dsname']}}'{Style.RESET_ALL} "
                                     f"( {Fore.MAGENTA}{threading.current_thread().name:{self.max_str_len_dict['thread_name']}}{Style.RESET_ALL} ) ")
            self.progressbar.refresh()
            self.gen_single_cam_gallery(fish_dsname)
        # ---------------------------------------------------------------------/