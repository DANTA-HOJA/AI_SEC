import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import pandas as pd
from colorama import Back, Fore, Style

from ...data import dname
from ...shared.baseobject import BaseObject
from ...shared.utils import create_new_dir, formatter_padr0
from ..processeddatainstance import ProcessedDataInstance
from .utils import gen_crop_img, gen_dataset_file_name_dict
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
        
        self._set_crop_dir_name()
        self._set_clustered_df()
        self._set_id2dataset_dict()
        self._set_dst_root()
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """ Set below attributes
            - `self.palmskin_result_name`: str
            - `self.cluster_desc`: str
        """
        """ [data_processed] """
        self.palmskin_result_name: str = self.config["data_processed"]["palmskin_result_name"]
        self.cluster_desc: str = self.config["data_processed"]["cluster_desc"]
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
            dataset_cropped.joinpath(self.cluster_desc.split("_")[-1],
                                     self._processed_di.instance_name,
                                     os.path.splitext(self.palmskin_result_name)[0])
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        
        # create necessary dir
        create_new_dir(self.dst_root.joinpath("test"))
        create_new_dir(self.dst_root.joinpath("train"))
        create_new_dir(self.dst_root.joinpath("valid"))
        
        # checking
        self._check_if_any_crop_dir_exists()
        self._processed_di.check_palmskin_images_condition(config)
        
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
                self.crop_single_image(palmskin_result_file)
                self._pbar.update(main_task, advance=1)
                self._pbar.refresh()
        
        # count dir
        self._cli_out.divide()
        for dir in ["test", "train", "valid"]:
            self._cli_out.write(f"{dir}: {len(list(self.dst_root.joinpath(dir).glob('*')))}")
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
            if replace: # (config varname TBA)
                self._cli_out.write(f"Deleting {len(existing_crop_dir)} '{self.crop_dir_name}' directories... ")
                for dir in existing_crop_dir: shutil.rmtree(dir)
                self._cli_out.write(f"{Fore.GREEN}{Back.BLACK} Done! {Style.RESET_ALL}")
            else:
                raise FileExistsError(f"{Fore.YELLOW}{Back.BLACK} To re-crop the images, "
                                      f"set `config.replace` = True {Style.RESET_ALL}\n") # WARNING: config not implemented
        # ---------------------------------------------------------------------/


    def crop_single_image(self, img_path:Path):
        """
        """
        img = cv2.imread(str(img_path))
        
        """ Extract info """
        fish_id, fish_pos = dname.get_dname_sortinfo(img_path)
        fish_dsname = f"fish_{fish_id}_{fish_pos}"
        fish_dataset = self.id2dataset_dict[fish_id]
        
        # copy file
        dsname_dir = self.dst_root.joinpath(fish_dataset, fish_dsname)
        cp_file = dsname_dir.joinpath(f"{fish_dsname}.tiff")
        if not cp_file.exists():
            create_new_dir(dsname_dir)
            shutil.copy(img_path, cp_file)
        
        # create `crop_dir`
        crop_dir = dsname_dir.joinpath(self.crop_dir_name)
        create_new_dir(crop_dir)
        
        # cropping
        crop_img_list = gen_crop_img(img, self.config)
        
        """ Sub Task """
        sub_task = self._pbar.add_task("[cyan][TBA]: ", total=len(crop_img_list))
        
        for i, cropped_img in enumerate(crop_img_list):
            
            cropped_name = f"{fish_dsname}_crop_{i:{formatter_padr0(crop_img_list)}}"
            
            self._pbar.update(sub_task, description=f"[cyan]{cropped_name} : ")
            self._pbar.refresh()
            
            save_path = crop_dir.joinpath(f"{cropped_name}.tiff")
            cv2.imwrite(str(save_path), cropped_img)
            
            self._pbar.update(sub_task, advance=1)
            self._pbar.refresh()
        
        self._pbar.remove_task(sub_task)
        # ---------------------------------------------------------------------/