import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

from tqdm.auto import tqdm

from ..ij.zfij import ZFIJ
from ..processeddatainstance import ProcessedDataInstance
from ...shared.clioutput import CLIOutput

from ...assert_fn import *
# -----------------------------------------------------------------------------/


class BrightfieldUNetAreaMeter():


    def __init__(self, zfij_instance:ZFIJ=None, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        # Initialize `Fiji` ( This will change the working directory to where the `JVM` exists. )
        if zfij_instance:
            self._zfij = zfij_instance
        else:
            self._zfij = ZFIJ(display_on_CLI)
        
        self.processed_data_instance = ProcessedDataInstance()
        self._cli_out = CLIOutput(display_on_CLI, logger_name="UNet Area Meter (Brightfield)")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------/



    def _set_attrs(self, config_file:Union[str, Path]):
        """
        """
        self.processed_data_instance.set_attrs(config_file)
        
        self.analyze_param_dict = \
            self.processed_data_instance.brightfield_processed_config["param"]
        # ---------------------------------------------------------------------/



    def run(self, config_file:Union[str, Path]="0.3.analyze_brightfield.toml"):
        """
        """
        self._cli_out.divide()
        self._set_attrs(config_file)
        
        dname_dirs = self.processed_data_instance.brightfield_processed_dname_dirs_dict.values()
        
        self._cli_out.divide()
        with tqdm(total=len(dname_dirs), desc=f"[ {self._cli_out.logger_name} ] : ") as pbar:
            
            for dname_dir in dname_dirs:
                self._single_unet_area_measurement(dname_dir)
                pbar.update(1)
                pbar.refresh()
        
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