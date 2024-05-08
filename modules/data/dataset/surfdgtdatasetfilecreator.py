import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import pandas as pd
from colorama import Back, Fore, Style

from ...shared.baseobject import BaseObject
from ...shared.utils import create_new_dir, exclude_tmp_paths
from .. import dname
from ..processeddatainstance import ProcessedDataInstance
from . import dsname
from .utils import drop_too_dark, gen_dataset_file_name_dict
# -----------------------------------------------------------------------------/


class SurfDGTDatasetFileCreator(BaseObject):

    def __init__(self, processed_data_instance:ProcessedDataInstance=None,
                 display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("SurfDGT Dataset File Creator")
        
        if processed_data_instance:
            self._processed_di = processed_data_instance
        else:
            self._processed_di = ProcessedDataInstance()
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.palmskin_result_name: str # self._set_config_attrs()
        self.cluster_desc: str # self._set_config_attrs()
        self.crop_dir_name: str # self._set_crop_dir_name()
        self.clustered_df: pd.DataFrame # self._set_clustered_df()
        self.id2area_dict: Dict[int, float] # self._set_id2area_dict()
        self.id2cls_dict: Dict[int, str] # self._set_id2cls_dict()
        self.dataset_file: Path # self._set_dataset_file()
        
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]):
        """
        """
        super()._set_attrs(config)
        self._processed_di.parse_config(config)
        
        self._set_crop_dir_name()
        self._set_clustered_df()
        self._set_id2area_dict()
        self._set_id2cls_dict()
        self._set_src_root()
        self._set_dataset_file()
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """ Set below attributes
            >>> self.palmskin_result_name: str
            >>> self.cluster_desc: str
        """
        """ [data_processed] """
        self.palmskin_result_name: str = self.config["data_processed"]["palmskin_result_name"]
        self.cluster_desc: str = self.config["data_processed"]["cluster_desc"]
        
        """ [param] """
        self.base_size: tuple[int, int] = self.config["param"]["base_size"]
        
        self.palmskin_result_name = os.path.splitext(self.palmskin_result_name)[0]
        # ---------------------------------------------------------------------/


    def _set_crop_dir_name(self):
        """
        """
        name_dict = gen_dataset_file_name_dict(self.config)
        self.crop_dir_name: str = \
            f"{name_dict['crop_size']}_{name_dict['shift_region']}"
        # ---------------------------------------------------------------------/


    def _set_clustered_df(self):
        """
        """
        try:
            clustered_file: Path = \
                self._processed_di.clustered_files_dict[self.cluster_desc]
        except KeyError:
            traceback.print_exc()
            print(f"{Fore.RED}{Back.BLACK} Can't find `{{{self.cluster_desc}}}_datasplit.csv`, "
                  f"please run `0.5.3.cluster_data.py` to create it. {Style.RESET_ALL}\n")
            sys.exit()
        
        # read CSV
        self.clustered_df: pd.DataFrame = \
            pd.read_csv(clustered_file, encoding='utf_8_sig')
        
        # add 'fish_id' column
        self.clustered_df["fish_id"] = \
            self.clustered_df["Brightfield"].apply(lambda x: dname.get_dname_sortinfo(x)[0])
        # ---------------------------------------------------------------------/


    def _set_id2area_dict(self):
        """
        """
        self.id2area_dict: dict = \
            { fish_id: area for fish_id, area in \
                zip(self.clustered_df["fish_id"], self.clustered_df["Trunk surface area, SA (um2)"])}
        # ---------------------------------------------------------------------/


    def _set_id2cls_dict(self):
        """
        """
        self.id2cls_dict: dict = \
            { fish_id: cls for fish_id, cls in \
                zip(self.clustered_df["fish_id"], self.clustered_df["class"])}
        # ---------------------------------------------------------------------/


    def _set_src_root(self):
        """
        """
        dataset_cropped: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
        
        self.src_root: Path = \
            dataset_cropped.joinpath(self.cluster_desc.split("_")[-1], # RND[xxx]
                                     self._processed_di.instance_name,
                                     self.palmskin_result_name,
                                     f"W{self.base_size[0]}_H{self.base_size[1]}")
        # ---------------------------------------------------------------------/


    def _set_dataset_file(self):
        """
        """
        cluster_desc_split = self.cluster_desc.split("_") # ['SURF3C', 'KMeansORIG', 'RND2022']
        classif_strategy: str = "_".join(cluster_desc_split[1:-1]) # 'KMeansORIG'
        
        dataset_file_name = "DS_SURFDGT"
        self.dataset_file: Path = \
            self.src_root.joinpath(classif_strategy, f"{dataset_file_name}.csv")
        
        if self.dataset_file.exists():
            raise FileExistsError(f"{Fore.RED}{Back.BLACK} target `dataset_file` already exists: "
                                  f"'{self.dataset_file}' {Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        
        # checking
        self._check_if_target_dirs_exist()
        
        # get images
        test_img_paths = list(self.src_root.glob(f"test/*/*.tiff"))
        test_img_paths = exclude_tmp_paths(test_img_paths)
        self._cli_out.write(f"test: {len(test_img_paths)} images")
        
        train_img_paths = list(self.src_root.glob(f"train/*/*.tiff"))
        train_img_paths = exclude_tmp_paths(train_img_paths)
        self._cli_out.write(f"train: {len(train_img_paths)} images")
        
        valid_img_paths = list(self.src_root.glob(f"valid/*/*.tiff"))
        valid_img_paths = exclude_tmp_paths(valid_img_paths)
        self._cli_out.write(f"valid: {len(valid_img_paths)} images")
        
        # collect all image
        img_paths = sorted(test_img_paths + train_img_paths + valid_img_paths,
                           key=dsname.get_dsname_sortinfo)
        
        """ Main Task """
        dataset_df: Union[None, pd.DataFrame] = None
        
        self._cli_out.divide()
        self._reset_pbar()
        with self._pbar:
            
            # add task to `self._pbar`
            task_desc = f"[yellow][ {self._cli_out.logger_name} ] : "
            task = self._pbar.add_task(task_desc, total=len(img_paths))
            
            for path in img_paths:
                
                """ Get Get `fish info` """
                rel_path: Path = path.relative_to(self.src_root)
                fish_dataset = rel_path.parts[0]
                parent_dsname = rel_path.parts[1] # without `_crop_[idx]`
                img_name = rel_path.stem
                
                parent_dsname_info = dsname.get_dsname_sortinfo(parent_dsname)
                img_name_info = dsname.get_dsname_sortinfo(img_name)
                if len(img_name_info) != len(parent_dsname_info):
                     img_size = "crop"
                else: img_size = "base"
                
                fish_id = img_name_info[0]
                fish_pos = img_name_info[1]
                fish_area = self.id2area_dict[fish_id]
                fish_class = self.id2cls_dict[fish_id]
                
                # update pbar
                dyn_desc = f"[yellow][ {self._cli_out.logger_name} ] ({fish_dataset}) {img_name} : "
                self._pbar.update(task, description=dyn_desc)
                self._pbar.refresh()
                
                """ preserve / discard """
                dark_ratio = "---"
                state = "preserve"
                if img_size == "crop":
                    
                    img = cv2.imread(str(path))
                    select, drop = drop_too_dark([img], self.config)
                    
                    if len(select) > 0:
                        assert len(drop) == 0, "Either `select` or `drop` needs to be empty"
                        dark_ratio = select[0][2]
                        state = "preserve"
                    
                    if len(drop) > 0:
                        assert len(select) == 0, "Either `select` or `drop` needs to be empty"
                        dark_ratio = drop[0][2]
                        state = "discard"
                
                """ Create `temp_dict` """
                temp_dict = {}
                # -------------------------------------------------------
                temp_dict["image_name"] = img_name
                temp_dict["area"] = fish_area
                temp_dict["class"] = fish_class
                temp_dict["image_size"] = img_size
                # -------------------------------------------------------
                temp_dict["parent (dsname)"] = parent_dsname
                temp_dict["fish_id"] = fish_id
                temp_dict["fish_pos"] = fish_pos
                # -------------------------------------------------------
                temp_dict["dataset"] = fish_dataset
                temp_dict["dark_ratio"] = dark_ratio
                temp_dict["state"] = state
                # -------------------------------------------------------
                temp_dict["path"] = rel_path
                # -------------------------------------------------------
                temp_df = pd.DataFrame(temp_dict, index=[0])
                
                # add to `dataset_df`
                if dataset_df is None: dataset_df = temp_df.copy()
                else: dataset_df = pd.concat([dataset_df, temp_df], ignore_index=True)
                
                self._pbar.update(task, advance=1)
                self._pbar.refresh()
        
        """ Save `dataset_xlsx` """
        self._cli_out.divide()
        self._cli_out.write("Saving `dataset_file`... ")
        create_new_dir(self.dataset_file.parent)
        dataset_df.to_csv(self.dataset_file, encoding='utf_8_sig', index=False)
        self._cli_out.write(f"{Fore.GREEN}{Back.BLACK} Done! {Style.RESET_ALL}")
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _check_if_target_dirs_exist(self):
        """
        """
        for dir in ["test", "train", "valid"]:
            if not self.src_root.joinpath(dir).exists():
                raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find directories, "
                                        f"run `1.1.crop_images.py` before create dataset file. {Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/