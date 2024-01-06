import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

from ...assert_fn import assert_lifname_split_in_4_part
from ...shared.baseobject import BaseObject
from ..ij.zfij import ZFIJ
from .lifnamechecker import LIFNameChecker
from .utils import scan_lifs_under_dir
# -----------------------------------------------------------------------------/


class BatchLIFNameChecker(BaseObject):

    def __init__(self, zfij_instance:ZFIJ=None, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Check Lif Name")
        self._lif_name_checker = LIFNameChecker()
        
        # Initialize `Fiji`
        if zfij_instance:
            self._zfij = zfij_instance
        else:
            self._zfij = ZFIJ(display_on_CLI)
        
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
        self._set_src_root()
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        self.nasdl_type    = self.config["data_nasdl"]["type"]
        self.nasdl_batches = self.config["data_nasdl"]["batches"]
        # ---------------------------------------------------------------------/


    def _set_src_root(self):
        """
        """
        self.src_root = self._path_navigator.raw_data.get_lif_scan_root(self.config, self._cli_out)
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """ Actions
        1. Load config
        2. Scan `LIF` files
        3. Check LIF name

        Args:
            config (Union[str, Path]): a toml file.
        """
        """ STEP 1. Load config """
        super().run(config)
        
        """ STEP 2. Scan `LIF` files """
        lif_paths = scan_lifs_under_dir(self.src_root, self.nasdl_batches, self._cli_out)
        
        """ STEP 3. Check LIF name """
        for i, lif_path in enumerate(lif_paths):
            
            self._cli_out.write(f"Processing ... {i+1}/{len(lif_paths)}")
            self._cli_out.write(f"LIF_FILE : '{lif_path}'")
            
            
            """ Normalize LIF name """
            lif_name = lif_path.split(os.sep)[-1].split(".")[0]
            lif_name_list = re.split(" |_|-", lif_name)
            assert_lifname_split_in_4_part(lif_name_list, lif_name)
            lif_name = "_".join(lif_name_list)
            
            """ Get number of images in LIF file """
            self._zfij.imageReader.setId(lif_path)
            series_cnt = self._zfij.imageReader.getSeriesCount()
            
            
            for idx in range(series_cnt):
                
                series_num = idx+1
                
                """ Read image """
                self._zfij.run("Bio-Formats Importer", f"open='{lif_path}' color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT series_{series_num}")
                img = self._zfij.ij.WindowManager.getCurrentImage() # get image, <java class 'ij.ImagePlus'>
                img.hide()
                
                """ Get image name """
                image_name = str(img.getProp("Image name"))
                
                
                """ BrightField_RAW """
                if self.nasdl_type == "BrightField_RAW":
                    if "Before_20221109" in lif_path:
                        self._lif_name_checker.check_image_name(image_name, "old_bf")
                    else: 
                        self._lif_name_checker.check_image_name(image_name, "new_bf")
                
                """ PalmSkin_RAW """
                if self.nasdl_type == "PalmSkin_RAW":
                    if "Before_20221109" in lif_path:
                        self._lif_name_checker.check_image_name(image_name, "old_rgb")
                    else: 
                        self._lif_name_checker.check_image_name(image_name, "new_rgb")
                
                
                """ Concat LIF and image name """
                comb_name = f"{lif_name} - {image_name}"
                
                """ Get image dimension """
                img_dimensions = img.getDimensions()
                self._cli_out.write(f"series {series_num:{len(str(series_cnt))}}/{series_cnt} : '{comb_name}' , "
                                    f"Dimensions : {img_dimensions} ( width, height, channels, slices, frames )")
                
                """ Print ERRORs after checking 'image name' """
                try:
                    self._cli_out.write(f"       ##### {self._lif_name_checker.check_dict['failed message']}")
                    self.run_pause("Use 'LAS X Office' to fix image name, press [ENTER] to continue.")
                except KeyError:
                    pass
                
                """ Close opened image """
                self._zfij.reset_all_window()
            
            self._cli_out.write("\n") # make CLI output prettier
        
        self._cli_out.write(" -- finished -- ")
        # ---------------------------------------------------------------------/


    def run_pause(self, msg:str):
        """
        """
        self._cli_out.divide()
        input(msg)
        self._cli_out.divide()
        # ---------------------------------------------------------------------/