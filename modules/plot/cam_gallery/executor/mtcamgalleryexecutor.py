import concurrent.futures
import os
import re
import sys
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple, Union

from tqdm.auto import tqdm

from ....data.dataset import dsname
from ..creator.camgallerycreator import CamGalleryCreator
from ..creator.mtcamgallerycreator import MtCamGalleryCreator
from .utils import divide_fish_dsname_in_group
# -----------------------------------------------------------------------------/


class MtCamGalleryExecutor(CamGalleryCreator):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super(CamGalleryCreator, self).__init__(display_on_CLI)
        self._cli_out._set_logger("Mt Cam Gallery Executor")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]):
        """
        """
        super(CamGalleryCreator, self)._set_attrs(config)
        self._set_history_dir()
        
        self._set_training_config_attrs()
        self._set_src_root()
        self._set_test_df()
        
        self._set_cam_result_root()
        self._set_cam_gallery_dir()
        
        self.max_str_len_dict: Dict[str, int] = {"fish_dsname": 0,
                                                 "thread_name": 0}
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


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super(CamGalleryCreator, self).run(config)
        
        fish_dsnames_group = self.divide_fish_dsnames()
        
        self._cli_out.divide()
        progressbars = \
            [ tqdm(total=len(fish_dsnames), desc=f"[ {self._cli_out.logger_name} ] ") \
                for fish_dsnames in fish_dsnames_group ]
        
        lock = Lock()
        with ThreadPoolExecutor(max_workers=self.worker) as t_pool:
            futures = [ t_pool.submit(
                            self.mt_task, *(fish_dsnames,
                                            self.max_str_len_dict,
                                            lock, progressbars[i],
                                            self._cli_out._display_on_CLI)
                        ) for i, fish_dsnames in enumerate(fish_dsnames_group) ]
            
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


    def divide_fish_dsnames(self) -> List[List[str]]:
        """
        """
        fish_dsnames = sorted(Counter(self.test_df["parent (dsname)"]).keys(),
                              key=dsname.get_dsname_sortinfo)
        # fish_dsnames = fish_dsnames[:32] # for debug
        
        # update `self.max_str_len_dict["fish_dsname"]`
        for fish_dsname in fish_dsnames:
            if len(fish_dsname) > self.max_str_len_dict["fish_dsname"]:
                self.max_str_len_dict["fish_dsname"] = len(fish_dsname)
        
        fish_dsnames_group = \
            divide_fish_dsname_in_group(fish_dsnames, self.worker)
        
        return fish_dsnames_group
        # ---------------------------------------------------------------------/


    def mt_task(self, sub_fish_dsnames, max_str_len_dict,
                lock, progressbar, display_on_CLI):
        """
        """
        with lock:
            mt_cam_gallery_creator = \
                MtCamGalleryCreator(max_str_len_dict, lock, progressbar,
                                    display_on_CLI)
        
        mt_cam_gallery_creator.mt_run(sub_fish_dsnames, self.config)
        # ---------------------------------------------------------------------/