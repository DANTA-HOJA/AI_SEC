import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from colorama import Back, Fore, Style

from ...data import dname
from ...dl.dataset.augmentation import crop_base_size
from ...shared.baseobject import BaseObject
from ...shared.utils import create_new_dir, formatter_padr0
from ..processeddatainstance import ProcessedDataInstance
from .utils import gen_crop_img_v2, gen_dataset_file_name_dict
# -----------------------------------------------------------------------------/


class ImageCropper(BaseObject):

    def __init__(self, processed_data_instance:ProcessedDataInstance=None,
                 display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Image Cropper")
        
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
        self.id2dataset_dict: Dict[int, str] # self._set_id2dataset_dict()
        
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]):
        """
        """
        super()._set_attrs(config)
        self._processed_di.parse_config(config)
        self._base_size_cropper = crop_base_size(*self.base_size)
        
        self._set_crop_dir_name()
        self._set_clustered_df()
        self._set_id2dataset_dict()
        self._set_dst_root()
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """ Set below attributes
            >>> self.palmskin_result_name: str
            >>> self.cluster_desc: str
            >>> self.base_size: tuple[int, int]
        """
        """ [data_processed] """
        self.palmskin_result_name: str = self.config["data_processed"]["palmskin_result_name"]
        self.cluster_desc: str = self.config["data_processed"]["cluster_desc"]
        
        """ [param] """
        self.base_size: tuple[int, int] = self.config["param"]["base_size"]
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


    def _set_id2dataset_dict(self):
        """
        """
        self.id2dataset_dict: dict = \
            { fish_id: dataset for fish_id, dataset in \
                zip(self.clustered_df["fish_id"], self.clustered_df["dataset"])}
        # ---------------------------------------------------------------------/


    def _set_dst_root(self):
        """
        """
        dataset_cropped: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
        
        self.dst_root: Path = \
            dataset_cropped.joinpath(self.cluster_desc.split("_")[-1], # RND[xxx]
                                     self._processed_di.instance_name,
                                     os.path.splitext(self.palmskin_result_name)[0],
                                     f"W{self.base_size[0]}_H{self.base_size[1]}")
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        
        # checking
        self._check_if_any_crop_dir_exists()
        self._processed_di.check_palmskin_images_condition(config)
        
        # create necessary dir
        create_new_dir(self.dst_root.joinpath("test"))
        create_new_dir(self.dst_root.joinpath("train"))
        create_new_dir(self.dst_root.joinpath("valid"))
        
        # get dict
        _, sorted_results_dict = \
            self._processed_di.get_sorted_results_dict("palmskin", self.palmskin_result_name)
        
        # get key
        palmskin_dnames = sorted(pd.concat([self.clustered_df["Palmskin Anterior (SP8)"],
                                            self.clustered_df["Palmskin Posterior (SP8)"]]),
                                 key=dname.get_dname_sortinfo)
        
        """ Main Task """
        self._cli_out.divide()
        self._reset_pbar()
        with self._pbar:
            
            # add task to `self._pbar`
            task_desc = f"[yellow][ {self._cli_out.logger_name} ] : "
            main_task = self._pbar.add_task(task_desc, total=len(palmskin_dnames))
            
            for palmskin_dname in palmskin_dnames:
                palmskin_result_file = sorted_results_dict.pop(palmskin_dname)
                self._create_single_basesize_imgset(palmskin_result_file)
                self._pbar.update(main_task, advance=1)
                self._pbar.refresh()
        
        # count dir
        self._cli_out.divide()
        for dir in ["test", "train", "valid"]:
            dsname_cnt = len(list(self.dst_root.joinpath(dir).glob("*")))
            img_cnt = len(list(self.dst_root.joinpath(dir).glob(f"**/{self.crop_dir_name}/*.tiff")))
            self._cli_out.write(f"{dir:5}, "
                                f"# of dsnames: {dsname_cnt:{len(str(len(palmskin_dnames)))}}, "
                                f"# of cropped images: {img_cnt}")
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _check_if_any_crop_dir_exists(self):
        """
        """
        replace = False #  TODO:  可以在 config 多加一個 replace 參數，選擇要不要重切

        # scan `crop_dir`
        existing_crop_dir = []
        for dir in ["test", "train", "valid"]:
            temp_list = list(self.dst_root.glob(f"{dir}/*/{self.crop_dir_name}"))
            if temp_list:
                self._cli_out.write(f"{Fore.YELLOW}{Back.BLACK} Detect {len(temp_list)} "
                                    f"'{self.crop_dir_name}' directories in '{dir}' {Style.RESET_ALL}")
            existing_crop_dir.extend(temp_list)

        if existing_crop_dir:
            if replace: # (config varname TBD)
                self._cli_out.write(f"Deleting {len(existing_crop_dir)} '{self.crop_dir_name}' directories... ")
                for dir in existing_crop_dir: shutil.rmtree(dir)
                self._cli_out.write(f"{Fore.GREEN}{Back.BLACK} Done! {Style.RESET_ALL}")
            else:
                raise FileExistsError(f"{Fore.YELLOW}{Back.BLACK} To re-crop the images, "
                                      f"set `config.replace` = True {Style.RESET_ALL}\n") # WARNING: config not implemented
        # ---------------------------------------------------------------------/


    def _create_single_hhc_imgset(self, img_path:Path):
        """ [deprecate] Actions:
            1. divide one `Anterior / Posterior` image into `UP(U) / Donw(D)` part
            2. crop image to dataset required format
            
            - hhc = abbr(horizontal half cut)
        """
        img = cv2.imread(str(img_path))
        img_dict = {
            "U": img[:512,:,:],
            "D": img[512:,:,:]
        }
        horizcut_task = self._pbar.add_task("[magenta][TBD]: ", total=len(img_dict))
        
        for part_abbr, part_img in img_dict.items():
            
            # extract info
            fish_id, fish_pos = dname.get_dname_sortinfo(img_path)
            fish_dataset = self.id2dataset_dict[fish_id]
            fish_dsname = f"fish_{fish_id}_{fish_pos}_{part_abbr}"
            self._pbar.update(horizcut_task, description=f"[magenta]{fish_dsname} : ")
            self._pbar.refresh()
            
            # save horizcut image
            dsname_dir = self.dst_root.joinpath(fish_dataset, fish_dsname)
            save_path = dsname_dir.joinpath(f"{fish_dsname}.tiff")
            if not save_path.exists():
                create_new_dir(dsname_dir)
                cv2.imwrite(str(save_path), part_img)
            
            self._pbar.update(horizcut_task, advance=1)
            self._pbar.refresh()
            
            # >>> Crop Task <<<
            if fish_dataset != "train":
                self._crop_single_image(part_img, dsname_dir, fish_dsname)
        
        self._pbar.remove_task(horizcut_task)
        # ---------------------------------------------------------------------/


    def _create_single_basesize_imgset(self, img_path:Path):
        """
        """
        img = cv2.imread(str(img_path))
        base_size_img = self._base_size_cropper(image=img)
        
        # extract info
        fish_id, fish_pos = dname.get_dname_sortinfo(img_path)
        fish_dataset = self.id2dataset_dict[fish_id]
        fish_dsname = f"fish_{fish_id}_{fish_pos}"
        
        # >>> Save `base_size_img` <<<
        dsname_dir = self.dst_root.joinpath(fish_dataset, fish_dsname)
        save_path = dsname_dir.joinpath(f"{fish_dsname}.tiff")
        if not save_path.exists():
            create_new_dir(dsname_dir)
            cv2.imwrite(str(save_path), base_size_img)
        
        # >>> Crop Task <<<
        self._crop_single_image(base_size_img, dsname_dir, fish_dsname)
        # ---------------------------------------------------------------------/


    def _crop_single_image(self, img:np.ndarray, dsname_dir:Path,
                            fish_dsname:str):
        """
        """
        # cropping
        crop_img_list = gen_crop_img_v2(img, self.config)
        crop_task = self._pbar.add_task("[cyan][TBD]: ", total=len(crop_img_list))
        
        # create `crop_dir`
        crop_dir = dsname_dir.joinpath(self.crop_dir_name)
        create_new_dir(crop_dir)
        
        for i, cropped_img in enumerate(crop_img_list):
            
            cropped_name = f"{fish_dsname}_crop_{i:{formatter_padr0(crop_img_list)}}"
            self._pbar.update(crop_task, description=f"[cyan]{cropped_name} : ")
            self._pbar.refresh()
            
            # save cropped images
            save_path = crop_dir.joinpath(f"{cropped_name}.tiff")
            cv2.imwrite(str(save_path), cropped_img)
            
            self._pbar.update(crop_task, advance=1)
            self._pbar.refresh()
        
        self._pbar.remove_task(crop_task)
        # ---------------------------------------------------------------------/