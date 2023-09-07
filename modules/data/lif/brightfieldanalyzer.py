import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from datetime import datetime

from .utils import scan_lifs_under_dir
from ..ij.zfij import ZFIJ
from ...shared.clioutput import CLIOutput
from ...shared.config import load_config, dump_config
from ...shared.pathnavigator import PathNavigator
from ...shared.utils import create_new_dir

from ...assert_fn import *



class BrightfieldAnalyzer():
    
    def __init__(self, zfij_instance:ZFIJ=None, display_on_CLI=True) -> None:
        """ Brightfield image analyzing
        """
        # Initialize `Fiji` ( This will change the working directory to where the `JVM` exists. )
        if zfij_instance:
            self._zfij = zfij_instance
        else:
            self._zfij = ZFIJ(display_on_CLI)
        
        self._path_navigator = PathNavigator()
        self._sn_digits = "02"
        
        """ CLI output """
        self._cli_out = CLIOutput(display_on_CLI, logger_name="Analyze Brightfield")
        
        self._reset_attrs()
        self._reset_single_img_attrs()
    
    
    
    def _reset_attrs(self):
        """
        """
        self.total_lif_file = 0
        self.lif_enum = 0
        self.brightfield_processed_dir = None
        self.mode = "NEW"
        self.analyze_param_dict = None
        self.log_writer = None
        self.alias_map = {}
    
    
    
    def run(self, config_file:Union[str, Path]="0.3.analyze_brightfield.toml"):
        """ Actions
        1. Load config
        2. Source: 
            1. Get `lif_scan_root`
            2. Scan `LIF` files
        3. Destination:
            1. Get `brightfield_processed_dir`
            2. Decide which `[param]` will be used
                - if mode == `NEW` -> Save the config to `brightfield_processed_dir`
                - if mode == `UPDATE` -> Load the `[param]` from existing config
            3. Open a `LOG_FILE`
        4. Analyze brightfield images
        5. Close `LOG_FILE` (3.2)
        6. Save `alias_map` as toml file under `brightfield_processed_dir`
        
        Args:
            config_file (Union[str, Path], optional): Defaults to `0.3.analyze_brightfield.toml`.
        """
        self._reset_attrs()
        self._cli_out.divide()
        
        """ STEP 1. Load config """
        config = load_config(config_file, cli_out=self._cli_out)
        nasdl_batches = config["data_nasdl"]["batches"]
        
        """ STEP 2. Source """
        lif_scan_root = self._path_navigator.raw_data.get_lif_scan_root(config, self._cli_out)
        lif_path_list = scan_lifs_under_dir(lif_scan_root, nasdl_batches, self._cli_out)
        self.total_lif_file = len(lif_path_list)
        
        """ STEP 3. Destination """
        """ 3.1. """
        self.brightfield_processed_dir = self._path_navigator.processed_data.get_processed_dir("brightfield", config, self._cli_out)
        if self.brightfield_processed_dir:
            self.mode = "UPDATE"
        else:
            instance_root = self._path_navigator.processed_data.get_instance_root(config, self._cli_out)
            reminder = config["data_processed"]["brightfield_reminder"]
            self.brightfield_processed_dir = instance_root.joinpath(f"{{{reminder}}}_BrightField_analyze")
            create_new_dir(self.brightfield_processed_dir)
        self._cli_out.write(f"Process Mode : {self.mode}")
        
        """ 3.2. """
        cp_config_path = self.brightfield_processed_dir.joinpath("brightfield_analyze_config.toml")
        if self.mode == "UPDATE":
            assert_file_exists(cp_config_path)
            self.analyze_param_dict = load_config(cp_config_path)["param"]
            self._cli_out.write(f"Parameters (load from): '{cp_config_path}'")
        else:
            self.analyze_param_dict = config["param"]
            dump_config(cp_config_path, config)
        
        """ 3.3. """
        time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        log_path = self.brightfield_processed_dir.joinpath(f"{{Logs}}_{{BrightfieldAnalyzer}}_{time_stamp}.log")
        self.log_writer = open(log_path, mode="w")
        
        """ STEP 4. Analyze brightfield images """
        for i, lif_path in enumerate(lif_path_list):
            # process single LIF file
            self.lif_enum = i+1
            self._single_lif_preprocess(lif_path)
        
        self.log_writer.write(f"{'-'*40}  finished  {'-'*40} \n")
        self._cli_out.write(" -- finished -- ")
        
        """ STEP 5. Close `LOG_FILE` """
        self.log_writer.close()
        
        """ STEP 6. Save `alias_map` """
        alias_map_path = self.brightfield_processed_dir.joinpath(f"brightfield_result_alias_map.toml")
        dump_config(alias_map_path, self.alias_map)
    
    
    
    def _reset_single_img_attrs(self):
        """
        """
        self.save_root = None
        self.metaimg_dir = None
        self.result_sn = 0
    
    
    
    def _single_lif_preprocess(self, lif_path:str):
        """
        """
        """ Display info """
        self._cli_out.write(f"Processing ... {self.lif_enum}/{self.total_lif_file}")
        self._cli_out.write(f"LIF_FILE : '{lif_path}'")
        
        """ Write `LOG_FILE` """
        self.log_writer.write(f"|{'-'*40}  Processing ... {self.lif_enum}/{self.total_lif_file}  {'-'*40} \n")
        self.log_writer.write(f"| \n")
        self.log_writer.write(f"|         LIF_FILE : {lif_path.split(os.sep)[-1]} \n")
        self.log_writer.write(f"| \n")
        
        
        """ Normalize LIF name """
        lif_name = lif_path.split(os.sep)[-1].split(".")[0]
        lif_name_list = re.split(" |_|-", lif_name)
        lif_name = "_".join(lif_name_list)
        
        """ Get number of images in LIF file """
        self._zfij.imageReader.setId(lif_path)
        series_cnt = self._zfij.imageReader.getSeriesCount()
        
        
        for idx in range(series_cnt):
            
            self._reset_single_img_attrs()
            series_num = idx+1
            
            self._zfij.run("Bio-Formats Importer", f"open='{lif_path}' color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT series_{series_num}")
            img = self._zfij.ij.WindowManager.getCurrentImage() # get image, <java class 'ij.ImagePlus'>
            img.hide()
            
            """ Get image name """
            image_name = str(img.getProp("Image name"))
            image_name_list = re.split(" |_|-", image_name)
            
            """ Normalize image name """
            if "Before_20221109" in lif_path:
                image_name_list.pop(3) # palmskin (old name only)
                image_name_list.pop(3) # [num]dpf (old name only)
                image_name_list.append("BF")
            image_name = "_".join(image_name_list)
            
            """ Concat LIF and image name """
            comb_name = f"{lif_name} - {image_name}"
            
            """ Print xy dimension info """
            img_dimensions = img.getDimensions()
            # dim_1
            dim1_length = img.getNumericProperty("Image #0|DimensionDescription #1|Length")
            dim1_elem = img.getNumericProperty("Image #0|DimensionDescription #1|NumberOfElements")
            dim1_unit = dim1_length/(dim1_elem-1)*(10**6)
            # dim_2
            dim2_length = img.getNumericProperty("Image #0|DimensionDescription #2|Length")
            dim2_elem = img.getNumericProperty("Image #0|DimensionDescription #2|NumberOfElements")
            dim2_unit = dim2_length/(dim2_elem-1)*(10**6)
            assert dim1_unit == dim2_unit, f"Voxel_X != Voxel_Y, Voxel_X, Voxel_Y = ({dim1_unit}, {dim2_unit}) micron/pixel"
            self._cli_out.write(f"series {series_num:{len(str(series_cnt))}}/{series_cnt} : '{comb_name}' , "
                                f"Dimensions : {img_dimensions} ( width, height, channels, slices, frames ), "
                                f"Voxel_X_Y : {dim1_unit:.4f} micron")
            
            """ Write `LOG_FILE` """
            self.log_writer.write(f"|-- processing ...  series {series_num:{len(str(series_cnt))}}/{series_cnt} in {self.lif_enum}/{self.total_lif_file} \n")
            self.log_writer.write(f"|         {comb_name} \n")
            self.log_writer.write(f"|         Dimensions : {img_dimensions} ( width, height, channels, slices, frames ), Voxel_X_Y : {dim1_unit:.4f} micron \n")
            
            """ Set `save_root`, `metaimg_dir` """
            if "delete" in comb_name:
                self.save_root = self.brightfield_processed_dir.joinpath("!~delete", comb_name)
            else:
                self.save_root = self.brightfield_processed_dir.joinpath(comb_name)
            assert_dir_not_exists(self.save_root)
            create_new_dir(self.save_root)
            self.metaimg_dir = self.save_root.joinpath("MetaImage")
            create_new_dir(self.metaimg_dir)
            
            
            """ Do preprocess """
            original_16bit = self.find_focused_plane(img, "original_16bit", self.metaimg_dir)
            
            micron_per_pixel = self.analyze_param_dict["micron_per_pixel"]
            self._zfij.run(img, "Set Scale...", f"distance=1 known={micron_per_pixel} unit=micron")
            
            convert_8bit = self.convert_to_8bit(img, "convert_8bit", self.metaimg_dir)
            cropped_BF = self.cropping(convert_8bit, "cropped_BF", self.save_root)
            auto_threshold = self.auto_threshold(cropped_BF, "auto_threshold", self.metaimg_dir)
            measured_mask = self.zf_measurement(auto_threshold, "measured_mask", self.metaimg_dir)
            
            """ Deal with ROI """
            self._zfij.roiManager.runCommand("Show All with labels")
            roi_cnt = int(self._zfij.roiManager.getCount())
            if roi_cnt > 0:
                """ ROI exists """
                if roi_cnt == 1:
                    """ success to get fish """
                    mix_img = self.average_fusion(cropped_BF, measured_mask, "cropped_BF--MIX", self.save_root)
                    self.save_roi()
                    self.save_measured_result()
                else:
                    """ logger and log_file """
                    self._cli_out.write(f"      ROI in RoiManager: {roi_cnt}")
                    self._cli_out.write(f"      #### ERROR : Number of ROI not = 1")
                    # Write Log
                    self.log_writer.write(f"|         number of ROI = {roi_cnt} \n")
                    self.log_writer.write("| #### ERROR : Number of ROI not = 1 \n")
            
            """ Close opened image """
            self._zfij.reset_all_window()
            self.log_writer.write(f"| \n") # make Log file looks better.
        
        self._cli_out.write("\n") # make CLI output prettier
        self.log_writer.write("\n\n\n")
    
    
    
    def _save_tif_with_SN(self, img, save_name:str, save_dir:Path) -> None:
        """
        """
        full_name = f"{self.result_sn:{self._sn_digits}}_{save_name}.tif"
        save_path = save_dir.joinpath(full_name)
        self._zfij.save_as_tiff(img, str(save_path))
        
        self.alias_map[f"{save_name}"] = str(save_path.relative_to(self.save_root))
        self.result_sn += 1
    
    
    
    def find_focused_plane(self, img, save_name:str, save_dir:Path):
        """
        Pick up focused slice if slices > 1
        Plugin ref : https://sites.google.com/site/qingzongtseng/find-focus
        Algorithm  : autofocus algorithm "Normalized variance"  (Groen et al., 1985; Yeo et al., 1993).
        """
        img_dimensions = img.getDimensions()
        slices = img_dimensions[3]
        
        if slices > 1:
            self._cli_out.write("      #### WARNING : Number of Slices > 1, run ' Find focused slices ' ") # WARNING:
            self.log_writer.write("| #### WARNING : Number of Slices > 1, run ' Find focused slices ' \n") # WARNING:
            self._zfij.run(img, "Find focused slices", "select=100 variance=0.000 select_only")
            img = self._zfij.ij.WindowManager.getCurrentImage()
            img.hide()
        
        """ Save as TIFF """
        self._save_tif_with_SN(img, save_name, save_dir)

        return img
    
    
    
    def convert_to_8bit(self, img, save_name:str, save_dir:Path):
        """
        """
        convert_8bit = img.duplicate()
        
        self._zfij.run("Conversions...", "scale weighted")
        self._zfij.run(convert_8bit, "8-bit", "")
        
        """ Save as TIFF """
        self._save_tif_with_SN(convert_8bit, save_name, save_dir)
        
        return convert_8bit
    
    
    
    def cropping(self, img, save_name:str, save_dir:Path):
        """
        """
        crop_rect = self.analyze_param_dict["crop_rect"]
        
        cropped_img = img.duplicate()
        cropped_img.setRoi(crop_rect["x"], crop_rect["y"], crop_rect["w"], crop_rect["h"])
        cropped_img = cropped_img.crop()
        
        """ Save as TIFF """
        self._save_tif_with_SN(cropped_img, save_name, save_dir)
        
        return cropped_img
    
    
    
    def auto_threshold(self, img, save_name:str, save_dir:Path):
        """
        """
        method = self.analyze_param_dict["auto_threshold"]
        
        thresholding = img.duplicate()
        self._zfij.run(thresholding, "Auto Threshold", f"method={method} white")
        
        thresholding = self.convert_to_mask(thresholding)
        
        """ Save as TIFF """
        self._save_tif_with_SN(thresholding, save_name, save_dir)
        
        return thresholding
    
    
    
    def convert_to_mask(self, img):
        """
        """
        mask = img.duplicate()
        
        self._zfij.ij.prefs.blackBackground = True
        self._zfij.run(mask, "Convert to Mask", "")
        
        return mask
    
    
    
    def zf_measurement(self, img, save_name:str, save_dir:Path):
        """
        """
        lower_bound = self.analyze_param_dict["measure_range"]["lower_bound"]
        upper_bound = self.analyze_param_dict["measure_range"]["upper_bound"]
        
        self._zfij.run("Set Measurements...", "area mean min feret's display redirect=None decimal=2")
        self._zfij.run(img, "Analyze Particles...", f"size={lower_bound}-{upper_bound} show=Masks display include add")
        measured_mask = self._zfij.ij.WindowManager.getCurrentImage()
        measured_mask.hide()
        
        measured_mask = self.convert_to_mask(measured_mask)
        
        """ Save as TIFF """
        self._save_tif_with_SN(measured_mask, save_name, save_dir)
        
        return measured_mask
    
    
    
    def average_fusion(self, image_1, image_2, save_name:str, save_dir:Path):
        """
        """
        avg_fusion = self._zfij.imageCalculator.run(image_1, image_2, "Average create")
        
        """ Save as TIFF """
        self._save_tif_with_SN(avg_fusion, save_name, save_dir)
        
        return avg_fusion
    
    
    
    def save_roi(self):
        """
        """
        save_path = self.metaimg_dir.joinpath("RoiSet.roi")
        self._zfij.roiManager.save(str(save_path))
        
        self.alias_map["RoiSet"] = str(save_path.relative_to(self.save_root))
    
    
    
    def save_measured_result(self):
        """
        """
        save_path = self.save_root.joinpath("AutoAnalysis.csv")
        self._zfij.save_as("Results", str(save_path))
        
        self.alias_map["AutoAnalysis"] = str(save_path.relative_to(self.save_root))