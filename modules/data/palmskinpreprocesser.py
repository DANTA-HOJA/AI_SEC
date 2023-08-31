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
        self.lif_enum = 0
        self.palmskin_processed_dir = None
        self.mode = "NEW"
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
                - if mode="NEW" -> Save the config to `palmskin_processed_dir`
                - if mode="UPDATE" -> Load the ["param"] from existing config
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
        self.palmskin_processed_dir = self._path_navigator.processed_data.get_processed_dir("palmskin", config, **self._display_kwargs)
        if self.palmskin_processed_dir:
            self.mode = "UPDATE"
        else:
            instance_root = self._path_navigator.processed_data.get_instance_root(config, **self._display_kwargs)
            reminder = config["data_processed"]["palmskin_reminder"]
            self.palmskin_processed_dir = instance_root.joinpath(f"{{{reminder}}}_PalmSkin_preprocess")
            create_new_dir(self.palmskin_processed_dir, display_on_CLI=False)
        self._logger.info(f"Process Mode : {self.mode}")
        
        """ 3.2. """
        cp_config_path = self.palmskin_processed_dir.joinpath("palmskin_preprocess_config.toml")
        if self.mode == "UPDATE":
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
            # process single LIF file
            self.lif_enum = i+1
            self._single_lif_preprocess(lif_path)
        
        self.log_writer.write(f"{'-'*40}  finished  {'-'*40} \n")
        self._logger.info(" -- finished -- ")
        
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
        """ Display info """
        self._logger.info(f"Processing ... {self.lif_enum}/{self.total_lif_file}")
        self._logger.info(f"LIF_FILE : {lif_path}")
        
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
            self.log_writer.write(f"|-- processing ...  series {series_num:{len(str(series_cnt))}}/{series_cnt} in {self.lif_enum}/{self.total_lif_file} \n")
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
            
            RGB_direct_max_zproj = self.direct_max_zproj(ch_list, "RGB_direct_max_zproj", self.save_root)
            
            ch_B_img_dict = self.channel_preprocess(ch_list[0], "B")
            ch_G_img_dict = self.channel_preprocess(ch_list[1], "G")
            ch_R_img_dict = self.channel_preprocess(ch_list[2], "R")
            
            RGB_m3d = self.merge_to_RGBstack(ch_R_img_dict["m3d"], ch_G_img_dict["m3d"], ch_B_img_dict["m3d"], "RGB_m3d", self.metaimg_dir)
            RGB_mm3d = self.merge_to_RGBstack(ch_R_img_dict["mm3d"], ch_G_img_dict["mm3d"], ch_B_img_dict["mm3d"], "RGB_mm3d", self.metaimg_dir)
            RGB_mm3d_kuwahara = self.merge_to_RGBstack(ch_R_img_dict["mm3d_kuwahara"], ch_G_img_dict["mm3d_kuwahara"], ch_B_img_dict["mm3d_kuwahara"], "RGB_mm3d_kuwahara", self.metaimg_dir)
            RGB_fusion = self.average_fusion(RGB_mm3d, RGB_mm3d_kuwahara, "RGB_fusion", self.save_root)
            RGB_fusion2Gray = self.RGB_to_Gray(RGB_fusion, "RGB_fusion2Gray", self.save_root)
            
            RGB_m3d_HE = self.histogram_equalization(RGB_m3d, "RGB_m3d_HE", self.metaimg_dir)
            RGB_mm3d_HE = self.histogram_equalization(RGB_mm3d, "RGB_mm3d_HE", self.metaimg_dir)
            RGB_mm3d_kuwahara_HE = self.histogram_equalization(RGB_mm3d_kuwahara, f"RGB_mm3d_kuwahara_HE", self.metaimg_dir)
            RGB_HE_fusion = self.average_fusion(RGB_mm3d_HE, RGB_mm3d_kuwahara_HE, "RGB_HE_fusion", self.save_root)
            RGB_HE_fusion2Gray = self.RGB_to_Gray(RGB_HE_fusion, "RGB_HE_fusion2Gray", self.save_root)
            
            """ Close opened image """
            self._zfij.reset_all_window()
            self.log_writer.write(f"| \n") # make Log file looks better.
        
        self._logger.info("\n") # make CLI output prettier
        self.log_writer.write("\n\n\n")
    
    
    
    def _save_tif_with_SN(self, img, save_name:str, save_dir:Path) -> None:
        """
        """
        full_name = f"{self.result_sn:{self._sn_digits}}_{save_name}.tif"
        save_path = save_dir.joinpath(full_name)
        self._zfij.save_as_tiff(img, str(save_path))
        
        self.alias_map[f"{save_name}"] = str(save_path.relative_to(self.save_root))
        self.result_sn += 1



    def merge_to_RGBstack(self, ch_R, ch_G, ch_B, save_name:str, save_dir:Path):
        """
        """
        RGBstack = self._zfij.rgbStackMerge.mergeChannels([ch_R, ch_G, ch_B], True)
        self._zfij.rgbStackConverter.convertToRGB(RGBstack)
        
        """ Save as TIFF """
        self._save_tif_with_SN(RGBstack, save_name, save_dir)
        
        return RGBstack



    def z_projection(self, img, method:str, save_name:str, save_dir:Path):
        """ 

        Args:
            method (_type_): "avg", "min", "max", "sum", "sd" or "median".
        """
        z_proj = self._zfij.zProjector.run(img, method)
        
        """ Save as TIFF """
        self._save_tif_with_SN(z_proj, save_name, save_dir)
        
        return z_proj



    def direct_max_zproj(self, img_list, save_name:str, save_dir:Path):
        """
        """
        ch_B = img_list[0]
        ch_G = img_list[1]
        ch_R = img_list[2]
        
        ch_B_direct = self.z_projection(ch_B, "max", "ch_B_direct", self.metaimg_dir)
        ch_G_direct = self.z_projection(ch_G, "max", "ch_G_direct", self.metaimg_dir)
        ch_R_direct = self.z_projection(ch_R, "max", "ch_R_direct", self.metaimg_dir)
        
        RGBstack = self.merge_to_RGBstack(ch_R_direct, ch_G_direct, ch_B_direct, save_name, save_dir)
        
        return RGBstack



    def median3D(self, img):
        """
        """
        radius_xyz = self.preprocess_param_dict["median3d_xyz"]
        
        median3D = img.duplicate()
        self._zfij.run(median3D, "Median 3D...", f"x={radius_xyz[0]} y={radius_xyz[1]} z={radius_xyz[2]}")
        
        return median3D



    def mean3D(self, img):
        """
        """
        radius_xyz = self.preprocess_param_dict["mean3d_xyz"]
        
        mean3D = img.duplicate()
        self._zfij.run(mean3D, "Mean 3D...", f"x={radius_xyz[0]} y={radius_xyz[1]} z={radius_xyz[2]}")
        
        return mean3D
    
    
    
    def kuwahara_filter(self, img, save_name:str, save_dir:Path):
        """
        """
        sampling = self.preprocess_param_dict["kuwahara_sampling"]
        
        kuwahara = img.duplicate()
        self._zfij.run(kuwahara, "Kuwahara Filter", f"sampling={sampling}")
        
        """ Save as TIFF """
        self._save_tif_with_SN(kuwahara, save_name, save_dir)
        
        return kuwahara
    
    
    
    def histogram_equalization(self, img, save_name:str, save_dir:Path):
        """
        """
        he = img.duplicate()
        self._zfij.run(he, "Enhance Contrast...", "saturated=0.35 equalize")
        
        """ Save as TIFF """
        self._save_tif_with_SN(he, save_name, save_dir)
        
        return he
    
    
    
    def average_fusion(self, image_1, image_2, save_name:str, save_dir:Path):
        """
        """
        avg_fusion = self._zfij.imageCalculator.run(image_1, image_2, "Average create")
        
        """ Save as TIFF """
        self._save_tif_with_SN(avg_fusion, save_name, save_dir)
        
        return avg_fusion
    
    
    
    def channel_preprocess(self, ch_img, ch_name:str):
        """
        """
        img_dict = {}
        
        m3d           = self.z_projection(self.median3D(ch_img), "max", f"ch_{ch_name}_m3d", self.metaimg_dir)
        mm3d          = self.z_projection(self.mean3D(m3d), "max", f"ch_{ch_name}_mm3d", self.metaimg_dir)
        mm3d_kuwahara = self.kuwahara_filter(mm3d, f"ch_{ch_name}_mm3d_kuwahara", self.metaimg_dir)
        fusion        = self.average_fusion(mm3d, mm3d_kuwahara, f"ch_{ch_name}_fusion", self.save_root)
        
        m3d_HE           = self.histogram_equalization(m3d, f"ch_{ch_name}_m3d_HE", self.metaimg_dir)
        mm3d_HE          = self.histogram_equalization(mm3d, f"ch_{ch_name}_mm3d_HE", self.metaimg_dir)
        mm3d_kuwahara_HE = self.histogram_equalization(mm3d_kuwahara, f"ch_{ch_name}_mm3d_kuwahara_HE", self.metaimg_dir)
        HE_fusion        = self.average_fusion(mm3d_HE, mm3d_kuwahara_HE, f"ch_{ch_name}_HE_fusion", self.save_root)
        
        img_dict["m3d"] = m3d
        img_dict["mm3d"] = mm3d
        img_dict["mm3d_kuwahara"] = mm3d_kuwahara
        img_dict["fusion"] = fusion
        img_dict["m3d_HE"] = m3d_HE
        img_dict["mm3d_HE"] = mm3d_HE
        img_dict["mm3d_kuwahara_HE"] = mm3d_kuwahara_HE
        img_dict["HE_fusion"] = HE_fusion
        
        return img_dict
    
    
    
    def RGB_to_Gray(self, rgb_img, save_name:str, save_dir:Path):
        """
        """
        gray_img = rgb_img.duplicate()
        self._zfij.run("Conversions...", "scale weighted")
        self._zfij.run(gray_img, "8-bit", "")
        
        """ Save as TIFF """
        self._save_tif_with_SN(gray_img, save_name, save_dir)
        
        return gray_img
        