import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

from .lifnamechecker import LIFNameChecker
from .utils import scan_lifs_under_dir
from ..ij.zfij import ZFIJ
from ...shared.clioutput import CLIOutput
from ...shared.pathnavigator import PathNavigator
from ...shared.utils import load_config

from ...assert_fn import assert_lifname_split_in_4_part



class BatchLIFNameChecker():
    
    def __init__(self, zfij_instance:ZFIJ=None, display_on_CLI=True) -> None:
        """
        """
        # Initialize `Fiji` ( This will change the working directory to where the `JVM` exists. )
        if zfij_instance:
            self._zfij = zfij_instance
        else:
            self._zfij = ZFIJ(display_on_CLI)
        
        self._path_navigator = PathNavigator()
        self._lif_name_checker = LIFNameChecker()
        
        """ CLI output """
        self._cli_out = CLIOutput(display_on_CLI, logger_name="Check Lif Name")
    
    
    
    def run(self, config_file:Union[str, Path]="0.1.check_lif_name.toml"):
        """ Actions
        1. Load config
        2. Source: 
            1. Get `lif_scan_root`
            2. Scan `LIF` files
        3. Check LIF name

        Args:
            config_file (Union[str, Path], optional): Defaults to `0.1.check_lif_name.toml`.
        """
        self._cli_out.divide()
        
        """ STEP 1. Load config """
        config = load_config(config_file, cli_out=self._cli_out)
        nasdl_type    = config["data_nasdl"]["type"]
        nasdl_batches = config["data_nasdl"]["batches"]
        
        """ STEP 2. Source """
        lif_scan_root = self._path_navigator.raw_data.get_lif_scan_root(config, self._cli_out)
        lif_path_list = scan_lifs_under_dir(lif_scan_root, nasdl_batches, self._cli_out)
        
        """ STEP 3. Check LIF name """
        for i, lif_path in enumerate(lif_path_list):
    
            self._cli_out.write(f"Processing ... {i+1}/{len(lif_path_list)}")
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
                if nasdl_type == "BrightField_RAW":
                    if "Before_20221109" in lif_path:
                        self._lif_name_checker.check_image_name(image_name, "old_bf")
                    else: 
                        self._lif_name_checker.check_image_name(image_name, "new_bf")
                
                """ PalmSkin_RAW """
                if nasdl_type == "PalmSkin_RAW":
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
                except KeyError:
                    pass
                
                """ Close opened image """
                self._zfij.reset_all_window()
            
            self._cli_out.write("\n") # make CLI output prettier
        
        self._cli_out.write(" -- finished -- ")