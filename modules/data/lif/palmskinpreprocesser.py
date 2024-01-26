import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

from ...assert_fn import *
from ...shared.baseobject import BaseObject
from ...shared.config import dump_config, load_config
from ...shared.utils import create_new_dir
from ..ij.zfij import ZFIJ
from .utils import scan_lifs_under_dir
# -----------------------------------------------------------------------------/


class PalmskinPreprocesser(BaseObject):

    def __init__(self, zfij_instance:ZFIJ=None, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Preprocess Palmskin")
        
        # Initialize `Fiji`
        if zfij_instance:
            self._zfij = zfij_instance
        else:
            self._zfij = ZFIJ(display_on_CLI)
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self._sn_digits:str = "02"
        
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]):
        """
        """
        super()._set_attrs(config)
        self._set_src_root()
        self._init_task_var()
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        self.nasdl_batches = self.config["data_nasdl"]["batches"]
        self.palmskin_reminder = self.config["data_processed"]["palmskin_reminder"]
        # ---------------------------------------------------------------------/


    def _set_src_root(self):
        """
        """
        self.src_root = self._path_navigator.raw_data.get_lif_scan_root(self.config, self._cli_out)
        # ---------------------------------------------------------------------/


    def _init_task_var(self):
        """
        """
        self.total_lif_file = 0
        self.palmskin_processed_dir = None
        self.preprocess_param_dict = {}
        self.log_writer = None
        self.lif_enum = 0
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """ Actions
        1. Load config
        2. Scan `LIF` files
        3. Set `palmskin_processed_dir`
        4. Set `preprocess_param_dict` (Decide which config `[param]` will be used)
            - if mode == `NEW` -> Save the config to `palmskin_processed_dir`
            - if mode == `UPDATE` -> Load the `[param]` from existing config in `palmskin_processed_dir`
        5. Open a `LOG_FILE`
        6. Preprocess palmskin images
        7. Close `LOG_FILE` ( opened in 5. )
        
        Args:
            config (Union[str, Path]): a toml file.
        """
        """ STEP 1. Load config """
        super().run(config)
        
        """ STEP 2. Scan `LIF` files """
        lif_paths = scan_lifs_under_dir(self.src_root, self.nasdl_batches, self._cli_out)
        self.total_lif_file = len(lif_paths)
        
        """ STEP 3. Set `palmskin_processed_dir` """
        instance_root = self._path_navigator.processed_data.get_instance_root(self.config, self._cli_out)
        self.palmskin_processed_dir = instance_root.joinpath(f"{{{self.palmskin_reminder}}}_PalmSkin_preprocess")
        if not self.palmskin_processed_dir.exists():
            create_new_dir(self.palmskin_processed_dir)
            self._cli_out.write(f"Process Mode : NEW")
        else:
            self._cli_out.write(f"Process Mode : UPDATE")
        
        """ STEP 4. Set `preprocess_param_dict` """
        palmskin_config = self.palmskin_processed_dir.joinpath("palmskin_preprocess_config.toml")
        if not palmskin_config.exists():
            self.preprocess_param_dict = self.config["param"]
            dump_config(palmskin_config, self.config)
        else:
            self.preprocess_param_dict = load_config(palmskin_config)["param"]
            self._cli_out.write(f"Preprocess Parameters (load from): '{palmskin_config}'")
        
        """ STEP 5. Open a `LOG_FILE` """
        time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        log_path = self.palmskin_processed_dir.joinpath(f"{{Logs}}_{{PalmskinPreprocesser}}_{time_stamp}.log")
        self.log_writer = open(log_path, mode="w")
        
        """ STEP 6. Preprocess palmskin images """
        self._cli_out.new_line()
        for i, lif_path in enumerate(lif_paths):
            # process single LIF file
            self.lif_enum = i+1 # (start from 1)
            self._single_lif_preprocess(lif_path)
        
        self.log_writer.write(f"{'-'*40}  finished  {'-'*40} \n")
        self._cli_out.write(" -- finished -- ")
        
        """ STEP 7. Close `LOG_FILE` """
        self.log_writer.close()
        # ---------------------------------------------------------------------/


    def _reset_single_img_attrs(self):
        """
        """
        self.dst_root = None
        self.metaimg_dir = None
        self.result_sn = 0
        # ---------------------------------------------------------------------/


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
            series_num = idx+1 # (start from 1)
            
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
            self._cli_out.write(f"series {series_num:{len(str(series_cnt))}}/{series_cnt} : '{comb_name}' , "
                                f"Dimensions : {img_dimensions} ( width, height, channels, slices, frames ), "
                                f"Voxel_Z : {voxel_z:.4f} micron")
            
            """ Write `LOG_FILE` """
            self.log_writer.write(f"|-- processing ...  series {series_num:{len(str(series_cnt))}}/{series_cnt} in {self.lif_enum}/{self.total_lif_file} \n")
            self.log_writer.write(f"|         {comb_name} \n")
            self.log_writer.write(f"|         Dimensions : {img_dimensions} ( width, height, channels, slices, frames ), Voxel_Z : {voxel_z:.4f} micron \n")
            
            """ Set `dst_root`, `metaimg_dir` """
            # dst_root
            if "del" in image_name_list[-1].lower(): # ( image name 尾有 delete 代表該照片品質不佳 )
                self.dst_root = self.palmskin_processed_dir.joinpath("+---delete", comb_name)
            else:
                self.dst_root = self.palmskin_processed_dir.joinpath(comb_name)
            assert_dir_not_exists(self.dst_root) # 已存在會直接 ERROR，防止誤寫其他二次分析結果
            create_new_dir(self.dst_root)
            # metaimg_dir
            self.metaimg_dir = self.dst_root.joinpath("MetaImage")
            create_new_dir(self.metaimg_dir)
            
            """ Do preprocess """
            self._zfij.run(img, "Set Scale...", "distance=2.2 known=1 unit=micron")
            ch_list = self._zfij.channelSplitter.split(img) # oreder = (B, G, R, BF)
            
            RGB_direct_max_zproj = self.direct_max_zproj(ch_list, "RGB_direct_max_zproj", self.dst_root)
            
            ch_B_img_dict = self.channel_preprocess(ch_list[0], "B")
            ch_G_img_dict = self.channel_preprocess(ch_list[1], "G")
            ch_R_img_dict = self.channel_preprocess(ch_list[2], "R")
            
            RGB_m3d = self.merge_to_RGBstack(ch_R_img_dict["m3d"], ch_G_img_dict["m3d"], ch_B_img_dict["m3d"], "RGB_m3d", self.metaimg_dir)
            RGB_mm3d = self.merge_to_RGBstack(ch_R_img_dict["mm3d"], ch_G_img_dict["mm3d"], ch_B_img_dict["mm3d"], "RGB_mm3d", self.metaimg_dir)
            RGB_mm3d_kuwahara = self.merge_to_RGBstack(ch_R_img_dict["mm3d_kuwahara"], ch_G_img_dict["mm3d_kuwahara"], ch_B_img_dict["mm3d_kuwahara"], "RGB_mm3d_kuwahara", self.metaimg_dir)
            RGB_fusion = self.average_fusion(RGB_mm3d, RGB_mm3d_kuwahara, "RGB_fusion", self.dst_root)
            RGB_fusion2Gray = self.RGB_to_Gray(RGB_fusion, "RGB_fusion2Gray", self.dst_root)
            
            RGB_m3d_HE = self.histogram_equalization(RGB_m3d, "RGB_m3d_HE", self.metaimg_dir)
            RGB_mm3d_HE = self.histogram_equalization(RGB_mm3d, "RGB_mm3d_HE", self.metaimg_dir)
            RGB_mm3d_kuwahara_HE = self.histogram_equalization(RGB_mm3d_kuwahara, f"RGB_mm3d_kuwahara_HE", self.metaimg_dir)
            RGB_HE_fusion = self.average_fusion(RGB_mm3d_HE, RGB_mm3d_kuwahara_HE, "RGB_HE_fusion", self.dst_root)
            RGB_HE_fusion2Gray = self.RGB_to_Gray(RGB_HE_fusion, "RGB_HE_fusion2Gray", self.dst_root)
            
            """ Close opened image """
            self._zfij.reset_all_window()
            self.log_writer.write(f"| \n") # make Log file looks better.
        
        self._cli_out.write("\n") # make CLI output prettier
        self.log_writer.write("\n\n\n")
        # ---------------------------------------------------------------------/


    def _save_tif_with_SN(self, img, save_name:str, save_dir:Path) -> None:
        """
        """
        full_name = f"{self.result_sn:{self._sn_digits}}_{save_name}.tif"
        save_path = save_dir.joinpath(full_name)
        self._zfij.save_as_tiff(img, str(save_path))
        
        self.result_sn += 1
        # ---------------------------------------------------------------------/


    def merge_to_RGBstack(self, ch_R, ch_G, ch_B, save_name:str, save_dir:Path):
        """
        """
        RGBstack = self._zfij.rgbStackMerge.mergeChannels([ch_R, ch_G, ch_B], True)
        self._zfij.rgbStackConverter.convertToRGB(RGBstack)
        
        """ Save as TIFF """
        self._save_tif_with_SN(RGBstack, save_name, save_dir)
        
        return RGBstack
        # ---------------------------------------------------------------------/


    def z_projection(self, img, method:str, save_name:str, save_dir:Path):
        """ 

        Args:
            method (_type_): "avg", "min", "max", "sum", "sd" or "median".
        """
        z_proj = self._zfij.zProjector.run(img, method)
        
        """ Save as TIFF """
        self._save_tif_with_SN(z_proj, save_name, save_dir)
        
        return z_proj
        # ---------------------------------------------------------------------/


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
        # ---------------------------------------------------------------------/


    def median3D(self, img):
        """
        """
        radius_xyz = self.preprocess_param_dict["median3d_xyz"]
        
        median3D = img.duplicate()
        self._zfij.run(median3D, "Median 3D...", f"x={radius_xyz[0]} y={radius_xyz[1]} z={radius_xyz[2]}")
        
        return median3D
        # ---------------------------------------------------------------------/


    def mean3D(self, img):
        """
        """
        radius_xyz = self.preprocess_param_dict["mean3d_xyz"]
        
        mean3D = img.duplicate()
        self._zfij.run(mean3D, "Mean 3D...", f"x={radius_xyz[0]} y={radius_xyz[1]} z={radius_xyz[2]}")
        
        return mean3D
        # ---------------------------------------------------------------------/


    def kuwahara_filter(self, img, save_name:str, save_dir:Path):
        """
        """
        sampling = self.preprocess_param_dict["kuwahara_sampling"]
        
        kuwahara = img.duplicate()
        self._zfij.run(kuwahara, "Kuwahara Filter", f"sampling={sampling}")
        
        """ Save as TIFF """
        self._save_tif_with_SN(kuwahara, save_name, save_dir)
        
        return kuwahara
        # ---------------------------------------------------------------------/


    def histogram_equalization(self, img, save_name:str, save_dir:Path):
        """
        """
        he = img.duplicate()
        self._zfij.run(he, "Enhance Contrast...", "saturated=0.35 equalize")
        
        """ Save as TIFF """
        self._save_tif_with_SN(he, save_name, save_dir)
        
        return he
        # ---------------------------------------------------------------------/


    def average_fusion(self, image_1, image_2, save_name:str, save_dir:Path):
        """
        """
        avg_fusion = self._zfij.imageCalculator.run(image_1, image_2, "Average create")
        
        """ Save as TIFF """
        self._save_tif_with_SN(avg_fusion, save_name, save_dir)
        
        return avg_fusion
        # ---------------------------------------------------------------------/


    def channel_preprocess(self, ch_img, ch_name:str):
        """
        """
        img_dict = {}
        
        m3d           = self.z_projection(self.median3D(ch_img), "max", f"ch_{ch_name}_m3d", self.metaimg_dir)
        mm3d          = self.z_projection(self.mean3D(m3d), "max", f"ch_{ch_name}_mm3d", self.metaimg_dir)
        mm3d_kuwahara = self.kuwahara_filter(mm3d, f"ch_{ch_name}_mm3d_kuwahara", self.metaimg_dir)
        fusion        = self.average_fusion(mm3d, mm3d_kuwahara, f"ch_{ch_name}_fusion", self.dst_root)
        
        m3d_HE           = self.histogram_equalization(m3d, f"ch_{ch_name}_m3d_HE", self.metaimg_dir)
        mm3d_HE          = self.histogram_equalization(mm3d, f"ch_{ch_name}_mm3d_HE", self.metaimg_dir)
        mm3d_kuwahara_HE = self.histogram_equalization(mm3d_kuwahara, f"ch_{ch_name}_mm3d_kuwahara_HE", self.metaimg_dir)
        HE_fusion        = self.average_fusion(mm3d_HE, mm3d_kuwahara_HE, f"ch_{ch_name}_HE_fusion", self.dst_root)
        
        img_dict["m3d"] = m3d
        img_dict["mm3d"] = mm3d
        img_dict["mm3d_kuwahara"] = mm3d_kuwahara
        img_dict["fusion"] = fusion
        img_dict["m3d_HE"] = m3d_HE
        img_dict["mm3d_HE"] = mm3d_HE
        img_dict["mm3d_kuwahara_HE"] = mm3d_kuwahara_HE
        img_dict["HE_fusion"] = HE_fusion
        
        return img_dict
        # ---------------------------------------------------------------------/


    def RGB_to_Gray(self, rgb_img, save_name:str, save_dir:Path):
        """
        """
        gray_img = rgb_img.duplicate()
        self._zfij.run("Conversions...", "scale weighted")
        self._zfij.run(gray_img, "8-bit", "")
        
        """ Save as TIFF """
        self._save_tif_with_SN(gray_img, save_name, save_dir)
        
        return gray_img
        # ---------------------------------------------------------------------/