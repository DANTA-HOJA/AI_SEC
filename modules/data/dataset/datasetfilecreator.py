import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import pandas as pd
from colorama import Back, Fore, Style
from rich.progress import track

from ...shared.baseobject import BaseObject
from ...shared.utils import (create_new_dir, exclude_tmp_paths,
                             get_target_str_idx_in_list)
from .. import dname
from ..processeddatainstance import ProcessedDataInstance
from . import dsname
from .utils import drop_too_dark, gen_crop_img_v2, gen_dataset_file_name_dict
# -----------------------------------------------------------------------------/


class DatasetFileCreator(BaseObject):

    def __init__(self, processed_data_instance:ProcessedDataInstance=None,
                 display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Dataset File Creator")
        
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
        self.base_size: tuple[int, int] = tuple(self.config["param"]["base_size"])
        self.crop_size: tuple[int, int] = tuple([self.config["param"]["crop_size"]]*2)
        
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
        
        name_dict = gen_dataset_file_name_dict(self.config)
        dataset_file_name = "_".join(name_dict.values())
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
        
        img_paths = self._collect_and_check_target_images()
        
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
        self._cli_out.write(f"Save Dir: {self.dataset_file.parent}")
        dataset_df.to_csv(self.dataset_file, encoding='utf_8_sig', index=False)
        self._cli_out.write(f"{Fore.GREEN}{Back.BLACK} Done! {Style.RESET_ALL}")
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _collect_and_check_target_images(self):
        """
        """
        """Get base size images"""
        img_paths = list(self.src_root.glob(f"train/*/*.tiff"))
        self._cli_out.write(f"train, size={self.base_size}: {len(img_paths)} images")
        
        """Get number of crops"""
        tmp_img = cv2.imread(str(img_paths[0]))
        num_of_crops = len(gen_crop_img_v2(tmp_img, self.config))
        
        """Collect sub-crops in each set"""
        for dir in ["test", "train", "valid"]:
            
            tmp_list = list(self.src_root.glob(f"{dir}/*/{self.crop_dir_name}/*.tiff"))
            tmp_list = exclude_tmp_paths(tmp_list)
            
            # get number of fish in current set
            num_of_fish = len(self.clustered_df[(self.clustered_df["dataset"] == f"{dir}")])
            
            # check number of cropped images
            if len(tmp_list) == (num_of_fish*2*num_of_crops):
                self._cli_out.write(f"{dir}, size={self.crop_size}: {len(tmp_list)} images")
            else:
                raise ValueError(f"{Fore.RED} Number of sub-crops in '{dir}' set incorrect, "
                                 f"should be {(num_of_fish*num_of_crops)}, "
                                 f"but only {len(tmp_list)} detected. "
                                 "Please re-execute `1.1.crop_images.py` to fix the problem."
                                 f"{Style.RESET_ALL}\n")
            
            # add to list
            img_paths.extend(tmp_list)
        
        """Sort images"""
        img_paths = sorted(img_paths, key=dsname.get_dsname_sortinfo)
        
        """Check images"""
        read_failed = 0
        for img_path in track(img_paths,  transient=True,
                              description=f"[yellow]Check Image Condition: "):
            if cv2.imread(str(img_path)) is None:
                read_failed += 1
                target_str = get_target_str_idx_in_list(img_path.parts, f"W{self.base_size[0]}_H{self.base_size[1]}")
                self._cli_out.write(f"{Fore.RED}{Back.BLACK}Can't read '{img_path.parts[-1]}' "
                                    f"in '{img_path.parts[target_str+1]}' set {Style.RESET_ALL}")
        
        if read_failed == 0: self._cli_out.write(f"Check Image Condition: {Fore.GREEN}Passed{Style.RESET_ALL}")
        else: raise RuntimeError(f"{Fore.RED} Due to broken images, the process has been halted. "
                                 "Please re-execute `1.1.crop_images.py` to fix the problem."
                                 f"{Style.RESET_ALL}\n")
        
        return img_paths
        # ---------------------------------------------------------------------/
