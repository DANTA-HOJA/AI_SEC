import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

from ...assert_fn import assert_file_ext
from ...shared.baseobject import BaseObject
from ...shared.utils import create_new_dir
from ..ij.zfij import ZFIJ
from ..processeddatainstance import ProcessedDataInstance
from .utils import scan_lifs_under_dir
# -----------------------------------------------------------------------------/


class PalmskinManualROICreator(BaseObject):

    def __init__(self, zfij_instance:ZFIJ=None,
                 processed_data_instance:ProcessedDataInstance=None,
                 display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Process Palmskin ManualROI")
        
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
        
        self._sn_digits:str = "02"
        
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]):
        """
        """
        super()._set_attrs(config)
        self._processed_di.parse_config(config)
        
        self._set_src_root()
        self._init_task_var()
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        """ [data_nasdl] """
        self.nasdl_batches: list[str] = self.config["data_nasdl"]["batches"]
        
        """ [data_processed] """
        self.palmskin_result_name: str = self.config["data_processed"]["palmskin_result_name"]
        assert_file_ext(self.palmskin_result_name, ".tif")
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
        self.lif_enum = 0
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        
        # >>> STEP 1. Scan `LIF` files <<<
        lif_paths = scan_lifs_under_dir(self.src_root, self.nasdl_batches, self._cli_out)
        self.total_lif_file = len(lif_paths)
        
        # >>> STEP 2. Preprocess palmskin images <<<
        self._cli_out.new_line()
        for i, lif_path in enumerate(lif_paths):
            # process single LIF file
            self.lif_enum = i+1 # (start from 1)
            self._single_lif_preprocess(lif_path)
        
        self._cli_out.write(" -- finished -- ")
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
            
            try:
                # >>> 找不到 dname_dir 就跳過 <<<
                self._processed_di.palmskin_processed_dname_dirs_dict[comb_name]
            except KeyError:
                continue
            
            # >>> set single image vars <<<
            
            # `self.dst_root`
            self.dst_root = self._processed_di.palmskin_processed_dir.joinpath(comb_name)
            
            # `self.metaimg_dir`
            self.metaimg_dir = self.dst_root.joinpath("ManualROI")
            create_new_dir(self.metaimg_dir)
            
            # `self.result_sn`
            tmp_list = list(self.dst_root.glob("*.tif"))
            tmp_list.extend(list(self.dst_root.glob("MetaImage/*.tif")))
            found_list = sorted(tmp_list, key=lambda x: os.path.split(x)[1])
            self.result_sn = int(found_list[-1].parts[-1].split("_")[0])+1
            
            # `rel_path`
            rel_path, _ = \
                self._processed_di.get_sorted_results_dict("palmskin", self.palmskin_result_name)
            
            # image scale
            self._zfij.run(img, "Set Scale...", "distance=2.2 known=1 unit=micron")
            
            # >>> Do preprocess <<<
            
            if not self.metaimg_dir.joinpath("ManualROI.roi").exists():
                
                ch_list = self._zfij.channelSplitter.split(img)
                
                # >>> image1: '03_RGB_direct_max_zproj.tif' <<<
                RGB_direct_max_zproj = self.direct_max_zproj(ch_list)
                
                # >>> image2: '[self.result_sn]_[method]BF_direct.tif' <<<
                ch_BF = ch_list[3]
                BF_set = {}
                BF_set["minBF_direct"] = self.z_projection(ch_BF, "min")
                BF_set["sumBF_direct"] = self.z_projection(ch_BF, "sum")
                BF_set["stdBF_direct"] = self.z_projection(ch_BF, "sd")
                
                # >>> `image2` overlap on `image1`
                for BF_name, BF_img in BF_set.items():
                    BF_on_RGB = self.average_fusion(RGB_direct_max_zproj, BF_img, BF_name, self.metaimg_dir)
                    BF_on_RGB.show()
                self._zfij.run("Tile", "") # display settings
                
                # >>> select and save with roi <<<
                self._zfij.roiManager.runCommand("Show All with labels")
                roi_cnt = int(self._zfij.roiManager.getCount())
                if roi_cnt == 1:
                    self.save_roi(self.metaimg_dir)
                    self.apply_roi(rel_path)
                else:
                    self._cli_out.write(f"      ROI in RoiManager: {roi_cnt}")
                    self._cli_out.write(f"      #### ERROR : Number of ROI not = 1")
                
            else:
                self.apply_roi(rel_path)
            
            """ Close opened image """
            self._zfij.reset_all_window()
        
        self._cli_out.write("\n") # make CLI output prettier
        # ---------------------------------------------------------------------/


    def direct_max_zproj(self, img_list):
        """
        """
        ch_B = img_list[0]
        ch_G = img_list[1]
        ch_R = img_list[2]
        
        ch_B_direct = self.z_projection(ch_B, "max")
        ch_G_direct = self.z_projection(ch_G, "max")
        ch_R_direct = self.z_projection(ch_R, "max")
        
        RGBstack = self.merge_to_RGBstack(ch_R_direct, ch_G_direct, ch_B_direct)
        
        return RGBstack
        # ---------------------------------------------------------------------/


    def z_projection(self, img, method:str):
        """

        Args:
            method (_type_): "avg", "min", "max", "sum", "sd" or "median".
        """
        z_proj = self._zfij.zProjector.run(img, method)
        
        return z_proj
        # ---------------------------------------------------------------------/


    def merge_to_RGBstack(self, ch_R, ch_G, ch_B):
        """
        """
        RGBstack = self._zfij.rgbStackMerge.mergeChannels([ch_R, ch_G, ch_B], True)
        self._zfij.rgbStackConverter.convertToRGB(RGBstack)
        
        return RGBstack
        # ---------------------------------------------------------------------/


    def average_fusion(self, image_1, image_2, save_name:str, save_dir:Path):
        """
        """
        avg_fusion = self._zfij.imageCalculator.run(image_1, image_2, "Average create")
        
        """ Save as TIFF """
        self._save_tif_with_SN(avg_fusion, save_name, save_dir)
        
        return avg_fusion
        # ---------------------------------------------------------------------/


    def _save_tif_with_SN(self, img, save_name:str, save_dir:Path) -> None:
        """
        """
        full_name = f"{self.result_sn:{self._sn_digits}}_{save_name}.tif"
        save_path = save_dir.joinpath(full_name)
        self._zfij.save_as_tiff(img, str(save_path))
        
        self.result_sn += 1
        # ---------------------------------------------------------------------/


    def save_roi(self, save_dir:Path):
        """
        """
        save_path = save_dir.joinpath("ManualROI.roi")
        self._zfij.roiManager.save(str(save_path))
        # ---------------------------------------------------------------------/


    def apply_roi(self, rel_path:str):
        """
        """
        img = self._zfij.ij.IJ.openImage(str(self.dst_root.joinpath(rel_path)))
        img.show()
        
        self._zfij.roiManager.runCommand("Show All with labels")
        roi_cnt = int(self._zfij.roiManager.getCount())
        if roi_cnt == 0:
            self._zfij.roiManager.open(str(self.metaimg_dir.joinpath("ManualROI.roi")))
        
        # clear ROI outside
        self._zfij.roiManager.select(0)
        self._zfij.ij.IJ.setBackgroundColor(0, 0, 0)
        self._zfij.ij.IJ.run(img, "Clear Outside", "")
        
        # save result
        orig_name = os.path.splitext(Path(rel_path).parts[-1])[0]
        new_name = f"{orig_name}.manualroi.tif"
        self._zfij.save_as_tiff(img, str(self.metaimg_dir.joinpath(new_name)))
        # ---------------------------------------------------------------------/