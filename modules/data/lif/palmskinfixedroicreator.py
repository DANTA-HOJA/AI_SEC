import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2

from ...assert_fn import assert_file_ext
from ...shared.baseobject import BaseObject
from ...shared.utils import create_new_dir
from ..processeddatainstance import ProcessedDataInstance
# -----------------------------------------------------------------------------/


class PalmskinFixedROICreator(BaseObject):

    def __init__(self, processed_data_instance:ProcessedDataInstance=None,
                 display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Process Palmskin FixedROI")
        
        if processed_data_instance:
            self._processed_di = processed_data_instance
        else:
            self._processed_di = ProcessedDataInstance()
        
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
        super()._set_attrs(config)
        self._processed_di.parse_config(config)
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        """ [data_processed] """
        self.palmskin_result_name: str = self.config["data_processed"]["palmskin_result_name"]
        assert_file_ext(self.palmskin_result_name, ".tif")
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        
        _, sorted_results_dict = \
            self._processed_di.get_sorted_results_dict("palmskin", self.palmskin_result_name)
        
        self._cli_out.new_line()
        self._reset_pbar()
        with self._pbar:
            
            # add task to `self._pbar`
            task_desc = f"[yellow][ {self._cli_out.logger_name} ] : "
            task = self._pbar.add_task(task_desc, total=len(sorted_results_dict))
            
            for fish_dname, result_path in sorted_results_dict.items():
                self._single_fixed_roi(fish_dname, result_path)
                self._pbar.update(task, advance=1)
                self._pbar.refresh()
        
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _single_fixed_roi(self, fish_dname:str, result_path:Path):
        """
        """
        img = cv2.imread(str(result_path))
        img = img[:,256:768,:]
        
        self.metaimg_dir = \
            self._processed_di.palmskin_processed_dir.joinpath(fish_dname, "FixedROI")
        create_new_dir(self.metaimg_dir)
        
        orig_name = os.path.splitext(Path(result_path).parts[-1])[0]
        new_name = f"{orig_name}.fixedroi.tif"
        cv2.imwrite(str(self.metaimg_dir.joinpath(new_name)), img)
        # ---------------------------------------------------------------------/