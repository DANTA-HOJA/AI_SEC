import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union, TextIO
from datetime import datetime

import toml
import tomlkit
from tomlkit.toml_document import TOMLDocument

from .ij.zfij import ZFIJ
from .lif.utils import scan_lifs_under_dir
from ..shared.logger import init_logger
from ..shared.utils import load_config, create_new_dir
from ..shared.pathnavigator import PathNavigator

from ..assert_fn import *



class PalmskinPreprocesser():

    def __init__(self, zfij_instance:ZFIJ=None) -> None:
        """ Palmskin image preprocessing
        """
        
        # Initialize `Fiji` ( This will change the working directory to where the `JVM` exists. )
        if zfij_instance:
            self._zfij = zfij_instance
        else:
            self._zfij = ZFIJ()
        
        """ Logger """
        self._logger = init_logger(r"Preprocess Palmskin")
        self._display_kwargs = {
            "display_on_CLI": True,
            "logger": self._logger
        }
        
        self._path_navigator = PathNavigator()
        self._sn_digits = "02"
        
        self._reset_attrs()
        self._reset_single_img_attrs()



    def _reset_attrs(self):
        """
        """
        self.total_lif_file = 0
        self.palmskin_processed_dir = None
        self.mode = "new"
        self.preprocess_param_dict = None
        self.log_writer = None
        self.alias_map = {}



    def run(self):
        """ Actions
        1. Load config: `0.2.preprocess_palmskin.toml`
        2. Source: 
            1. Get `lif_scan_root`
            2. Scan `LIF` files
        3. Destination:
            1. Get `palmskin_processed_dir`
            2. Decide which ["param"] will be used
                - if mode="new" -> Save the config to `palmskin_processed_dir`
                - if mode="update" -> Load the ["param"] from existing config
            3. Open a `LOG_FILE`
        4. Preprocess palmskin images
        5. Close `LOG_FILE` (3.2)
        6. Save `alias_map` as toml file under `palmskin_processed_dir`
        """
        self._reset_attrs()
        
        """ STEP 1. Load config """
        config = load_config("0.2.preprocess_palmskin.toml", **self._display_kwargs)
        nasdl_batches = config["data_nasdl"]["batches"]
        
        """ STEP 2. Source """
        lif_scan_root = self._path_navigator.raw_data.get_lif_scan_root(config, **self._display_kwargs)
        lif_path_list = scan_lifs_under_dir(lif_scan_root, nasdl_batches, logger=self._logger)
        self.total_lif_file = len(lif_path_list)
        
        """ STEP 3. Destination """
        """ 3.1. """
        try:
            self.palmskin_processed_dir = self._path_navigator.processed_data.get_processed_dir("palmskin", config, **self._display_kwargs)
            assert_dir_exists(self.palmskin_processed_dir)
            self.mode = "update"
        except FileNotFoundError:
            create_new_dir(self.palmskin_processed_dir, display_on_CLI=False)
        
        """ 3.2. """
        cp_config_path = self.palmskin_processed_dir.joinpath("palmskin_preprocess_config.toml")
        if self.mode == "update":
            assert_file_exists(cp_config_path)
            with open(cp_config_path, mode="r") as f_reader:
                self.preprocess_param_dict = toml.load(f_reader)["param"]
        else:
            self.preprocess_param_dict = config["param"]
            with open(cp_config_path, mode="w") as f_writer:
                tomlkit.dump(config, f_writer)
        
        """ 3.3. """
        time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        log_path = self.palmskin_processed_dir.joinpath(f"{{Logs}}_{{PalmskinPreprocesser}}_{time_stamp}.log")
        self.log_writer = open(log_path, mode="w")
        
        """ STEP 4. Preprocess palmskin images """
        for i, lif_path in enumerate(lif_path_list):
    
            self._logger.info(f"Processing ... {i+1}/{len(lif_path_list)}")
            
            self.log_writer.write("\n\n\n")
            self.log_writer.write(f"|{'-'*40}  Processing ... {i+1}/{len(lif_path_list)}  {'-'*40} \n")
            self.log_writer.write(f"| \n")
            self.log_writer.write(f"|         LIF_FILE : {lif_path.split(os.sep)[-1]} \n")
            self.log_writer.write(f"| \n")
            
            # process current LIF_FILE
            self._single_lif_preprocess(lif_path)
            self._logger.info("\n") # make CLI output looks better.

        self.log_writer.write(f"\n\n\n{'-'*40}  finished  {'-'*40} \n")
        self._logger.info(" -- finished --")
        
        """ STEP 5. Close `LOG_FILE` """
        self.log_writer.close()
        
        """ STEP 6. Save `alias_map` """
        alias_map_path = self.palmskin_processed_dir.joinpath(f"palmskin_result_alias_map.toml")
        with open(alias_map_path, mode="w") as f_writer:
            toml.dump(self.alias_map, f_writer)
    


    def _reset_single_img_attrs(self):
        """
        """
        self.save_root = None
        self.metaimg_dir = None
        self.result_sn = 0
    
    
    
    def _single_lif_preprocess(self, lif_path:str):
        """
        """
        self._logger.info(f'LIF_FILE : {lif_path}')
        
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
            img =  self._zfij.ij.WindowManager.getCurrentImage() # get image, <java class 'ij.ImagePlus'>
            img.hide()
            
            """ Get image name """
            image_name = str(img.getProp("Image name"))
            image_name_list = re.split(" |_|-", image_name)
            
            """ Normalize image name """
            if "Before_20221109" in lif_path:
                image_name_list.pop(3) # palmskin (old name only)
                image_name_list.pop(3) # [num]dpf (old name only)
                image_name_list.append("RGB")
            image_name = "_".join(image_name_list)
            
            """ Concat LIF and image name """
            comb_name = f"{lif_name} - {image_name}"
            
            """ Print z dimension info """
            img_dimensions = img.getDimensions()
            z_length = img.getNumericProperty("Image #0|DimensionDescription #6|Length")
            z_slice = img.getNumericProperty("Image #0|DimensionDescription #6|NumberOfElements")
            voxel_z = z_length/(z_slice-1)*(10**6)
            self._logger.info(f"series {series_num:{len(str(series_cnt))}}/{series_cnt} : '{comb_name}' , "
                              f"Dimensions : {img_dimensions} ( width, height, channels, slices, frames ), "
                              f"Voxel_Z : {voxel_z:.4f} micron")
            
            """ Write `LOG_FILE` """
            self.log_writer.write(f"|-- processing ...  series {series_num:{len(str(series_cnt))}}/{series_cnt} in {series_num}/{self.total_lif_file} \n")
            self.log_writer.write(f"|         {comb_name} \n")
            self.log_writer.write(f"|         Dimensions : {img_dimensions} ( width, height, channels, slices, frames ), Voxel_Z : {voxel_z:.4f} micron \n")
            
            """ Set `save_root`, `metaimg_dir` """
            if "delete" in comb_name:
                self.save_root = self.palmskin_processed_dir.joinpath("!~delete", comb_name)
            else:
                self.save_root = self.palmskin_processed_dir.joinpath(comb_name)
            assert_dir_not_exists(self.save_root)
            create_new_dir(self.save_root, display_on_CLI=False)
            self.metaimg_dir = self.save_root.joinpath("MetaImage")
            create_new_dir(self.metaimg_dir, display_on_CLI=False)
            
            
            """ Do preprocess """
            self._zfij.run(img, "Set Scale...", "distance=2.2 known=1 unit=micron")
            ch_list = self._zfij.channelSplitter.split(img)
            
            """ `preprocess_param_dict` """
            kuwahara_sampling = self.preprocess_param_dict["kuwahara_sampling"]
            
            RGB_direct_max_zproj = self.direct_max_zproj(ch_list, "RGB_direct_max_zproj", "RGB_direct_max_zproj")
            
            ch_B_img_dict = self.channel_preprocess(ch_list[0], "B")
            ch_G_img_dict = self.channel_preprocess(ch_list[1], "G")
            ch_R_img_dict = self.channel_preprocess(ch_list[2], "R")
            
            RGB_processed = self.merge_to_RGBstack(ch_R_img_dict["ch_R"], ch_G_img_dict["ch_G"], ch_B_img_dict["ch_B"])
            self._save_tif_with_SN(RGB_processed, "RGB_processed", self.metaimg_dir, "RGB")
            
            RGB_processed_kuwahara = self.merge_to_RGBstack(ch_R_img_dict["ch_R_kuwahara"], ch_G_img_dict["ch_G_kuwahara"], ch_B_img_dict["ch_B_kuwahara"])
            self._save_tif_with_SN(RGB_processed_kuwahara, f"RGB_processed_kuwahara{kuwahara_sampling}", self.metaimg_dir, "RGB_kuwahara")
            
            RGB_processed_HE = self.histogram_equalization(RGB_processed, "RGB_processed_HE", "RGB_HE")
            RGB_processed_kuwahara_HE = self.histogram_equalization(RGB_processed_kuwahara, f"RGB_processed_kuwahara{kuwahara_sampling}_HE", "RGB_kuwahara_HE")

            RGB_processed_fusion = self.average_fusion(RGB_processed, RGB_processed_kuwahara, "RGB_processed_fusion", "RGB_fusion")
            RGB_processed_fusion2Gray = self.RGB_to_Gray(RGB_processed_fusion, "RGB_processed_fusion2Gray", "RGB_fusion2Gray")
            
            RGB_processed_HE_fusion = self.average_fusion(RGB_processed_HE, RGB_processed_kuwahara_HE, "RGB_processed_HE_fusion", "RGB_HE_fusion")
            RGB_processed_HE_fusion2Gray = self.RGB_to_Gray(RGB_processed_HE_fusion, "RGB_processed_HE_fusion2Gray", "RGB_HE_fusion2Gray")
    
    
    
    def _save_tif_with_SN(self, img, save_name:str, save_dir:Path, alias_name:str) -> None:
        """
        """
        full_name = f"{self.result_sn:{self._sn_digits}}_{save_name}.tif"
        save_path = save_dir.joinpath(full_name)
        self._zfij.save_as_tiff(img, str(save_path))
        
        self.alias_map[f"{alias_name}"] = str(save_path.relative_to(self.save_root))
        self.result_sn += 1



    def merge_to_RGBstack(self, ch_R, ch_G, ch_B):
        """
        """
        RGBstack = self._zfij.rgbStackMerge.mergeChannels([ch_R, ch_G, ch_B], True)
        self._zfij.rgbStackConverter.convertToRGB(RGBstack)
        
        return RGBstack



    def direct_max_zproj(self, img_list, save_name:str, alias_name:str):
        """
        """
        ch_B = img_list[0]
        ch_G = img_list[1]
        ch_R = img_list[2]
        
        max_zproj_ch_B = self._zfij.zProjector.run(ch_B, "max")
        max_zproj_ch_G = self._zfij.zProjector.run(ch_G, "max")
        max_zproj_ch_R = self._zfij.zProjector.run(ch_R, "max")
        
        RGBstack = self.merge_to_RGBstack(max_zproj_ch_R, max_zproj_ch_G, max_zproj_ch_B)
        
        self._save_tif_with_SN(RGBstack, save_name, self.save_root, alias_name)
        
        return RGBstack



    def median_R1_and_mean3D_R2(self, img, save_name:str, alias_name:str):
        """
        """
        """ Median Filter """
        median_r1 = img.duplicate()
        self._zfij.run(median_r1, "Median...", "radius=1 stack")
        
        """ Mean Filter 3D """
        median_r1_mean3D_r2 = median_r1.duplicate()
        self._zfij.run(median_r1_mean3D_r2, "Mean 3D...", "x=2 y=2 z=2")
        
        """ Maximum Projection """
        median_r1_mean3D_r2_Zproj_max = self._zfij.zProjector.run(median_r1_mean3D_r2, "max") # 'method' is "avg", "min", "max", "sum", "sd" or "median".
        
        """ Save as TIFF """
        self._save_tif_with_SN(median_r1_mean3D_r2_Zproj_max, save_name, self.metaimg_dir, alias_name)
        
        return median_r1_mean3D_r2_Zproj_max
    
    
    
    def kuwahara_filter(self, img, sampling:int, save_name:str, alias_name:str):
        """
        """
        kuwahara = img.duplicate()
        self._zfij.run(kuwahara, "Kuwahara Filter", f"sampling={sampling}")
        
        """ Save as TIFF """
        self._save_tif_with_SN(kuwahara, save_name, self.metaimg_dir, alias_name)
        
        return kuwahara
    
    
    
    def histogram_equalization(self, img, save_name:str, alias_name:str):
        """
        """
        he = img.duplicate()
        self._zfij.run(he, "Enhance Contrast...", "saturated=0.35 equalize")
        
        """ Save as TIFF """
        self._save_tif_with_SN(he, save_name, self.metaimg_dir, alias_name)
        
        return he
    
    
    
    def average_fusion(self, image_1, image_2, save_name:str, alias_name:str):
        """
        """
        avg_fusion = self._zfij.imageCalculator.run(image_1, image_2, "Average create")
        
        """ Save as TIFF """
        self._save_tif_with_SN(avg_fusion, save_name, self.save_root, alias_name)
        
        return avg_fusion
    
    
    
    def channel_preprocess(self, ch_img, ch_name:str):
        """
        """
        kuwahara_sampling = self.preprocess_param_dict["kuwahara_sampling"]
        img_dict = {}
        
        processed             = self.median_R1_and_mean3D_R2(ch_img, f"{ch_name}_processed", f"ch_{ch_name}")
        processed_HE          = self.histogram_equalization(processed, f"{ch_name}_processed_HE", f"ch_{ch_name}_HE")
        
        processed_kuwahara    = self.kuwahara_filter(processed, kuwahara_sampling, f"{ch_name}_processed_kuwahara{kuwahara_sampling}", f"ch_{ch_name}_kuwahara")
        processed_kuwahara_HE = self.histogram_equalization(processed_kuwahara, f"{ch_name}_processed_kuwahara{kuwahara_sampling}_HE", f"ch_{ch_name}_kuwahara_HE")
        
        processed_fusion      = self.average_fusion(processed, processed_kuwahara, f"{ch_name}_processed_fusion", f"ch_{ch_name}_fusion")
        processed_HE_fusion   = self.average_fusion(processed_HE, processed_kuwahara_HE, f"{ch_name}_processed_HE_fusion", f"ch_{ch_name}_HE_fusion")
        
        img_dict[f"ch_{ch_name}"] = processed
        img_dict[f"ch_{ch_name}_HE"] = processed_HE
        img_dict[f"ch_{ch_name}_kuwahara"] = processed_kuwahara
        img_dict[f"ch_{ch_name}_kuwahara_HE"] = processed_kuwahara_HE
        img_dict[f"ch_{ch_name}_fusion"] = processed_fusion
        img_dict[f"ch_{ch_name}_HE_fusion"] = processed_HE_fusion
        
        return img_dict
    
    
    
    def RGB_to_Gray(self, rgb_img, save_name:str, alias_name:str):
        """
        """
        gray_img = rgb_img.duplicate()
        self._zfij.run("Conversions...", "scale weighted")
        self._zfij.run(gray_img, "8-bit", "")
        
        """ Save as TIFF """
        self._save_tif_with_SN(gray_img, save_name, self.save_root, alias_name)
        
        return gray_img
        