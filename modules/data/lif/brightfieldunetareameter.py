import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

from ...assert_fn import *
from ...shared.baseobject import BaseObject
from ..ij.zfij import ZFIJ
from ..processeddatainstance import ProcessedDataInstance
# -----------------------------------------------------------------------------/


class BrightfieldUNetAreaMeter(BaseObject):

    def __init__(self, zfij_instance:ZFIJ=None, processed_data_instance=None,
                 display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Measure UNet Area (Brightfield)")
        
        # Initialize `Fiji`
        if zfij_instance:
            self._zfij = zfij_instance
        else:
            self._zfij = ZFIJ(display_on_CLI)
        
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
        
        self.analyze_param_dict = \
            self._processed_di.brightfield_processed_config["param"]
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        pass
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        
        dname_dirs = self._processed_di.brightfield_processed_dname_dirs_dict.values()
        
        self._cli_out.divide()
        self._reset_pbar()
        with self._pbar:
            
            # add task to `self._pbar`
            task_desc = f"[yellow][ {self._cli_out.logger_name} ] : "
            task = self._pbar.add_task(task_desc, total=len(dname_dirs))
            
            for dname_dir in dname_dirs:
                self._single_unet_area_measurement(dname_dir)
                self._pbar.update(task, advance=1)
                self._pbar.refresh()
        
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _single_unet_area_measurement(self, dname_dir:Path):
        """
        """
        mask_path = dname_dir.joinpath("UNet_predict_mask.tif")
        
        img = self._zfij.ij.IJ.openImage(str(mask_path))
        img.hide()
        
        micron_per_pixel = self.analyze_param_dict["micron_per_pixel"]
        self._zfij.run(img, "Set Scale...", f"distance=1 known={micron_per_pixel} unit=micron")
        
        self.convert_to_mask(img)
        self.zf_measurement(img)
        
        self._zfij.roiManager.runCommand("Show All with labels")
        roi_cnt = int(self._zfij.roiManager.getCount())
        if roi_cnt == 1:
            self.save_measured_result(dname_dir)
        
        self._zfij.reset_all_window()
        # ---------------------------------------------------------------------/


    def convert_to_mask(self, img):
        """
        """
        self._zfij.ij.prefs.blackBackground = True
        self._zfij.run(img, "Convert to Mask", "")
        # ---------------------------------------------------------------------/


    def zf_measurement(self, img):
        """
        """
        lower_bound = self.analyze_param_dict["measure_range"]["lower_bound"]
        upper_bound = self.analyze_param_dict["measure_range"]["upper_bound"]
        
        self._zfij.run("Set Measurements...", "area mean min feret's display redirect=None decimal=2")
        self._zfij.run(img, "Analyze Particles...", f"size={lower_bound}-{upper_bound} display include add")
        # ---------------------------------------------------------------------/


    def save_measured_result(self, dname_dir:Path):
        """
        """
        save_path = dname_dir.joinpath("UNetAnalysis.csv")
        self._zfij.save_as("Results", str(save_path))
        # ---------------------------------------------------------------------/