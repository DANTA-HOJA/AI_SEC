import json
import os
import re
import shutil
import sys
from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from colorama import Back, Fore, Style
from PIL import Image
from tomlkit.toml_document import TOMLDocument
from tqdm.auto import tqdm

from ....assert_fn import assert_0_or_1_history_dir
from ....data.dataset import dsname
from ....data.dataset.utils import parse_dataset_file_name
from ....shared.baseobject import BaseObject
from ....shared.config import load_config
from ....shared.utils import create_new_dir
from ...utils import (draw_predict_ans_on_image, draw_x_on_image,
                      get_mono_font, plot_with_imglist_auto_row)
# -----------------------------------------------------------------------------/


class CamGalleryCreator(BaseObject):

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        super().__init__(display_on_CLI)
        self._cli_out._set_logger("Cam Gallery Creator")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]):
        """
        """
        super()._set_attrs(config)
        self._set_history_dir()
        self._set_training_config_attrs()
        self._set_config_attrs_default_value()
        
        self._set_src_root()
        self._set_test_df()
        self._set_predict_ans_dict()
        self._set_cam_result_root()
        
        self._set_cam_gallery_dir()
        self._set_rank_dict()
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        """ [model_prediction] """
        self.model_time_stamp: str = self.config["model_prediction"]["time_stamp"]
        self.model_state: str = self.config["model_prediction"]["state"]
        
        """ [layout] """
        self.column: int = self.config["layout"]["column"]
        
        """ [draw.drop_image.line] """
        self.line_color: list = self.config["draw"]["drop_image"]["line"]["color"]
        self.line_width: int = self.config["draw"]["drop_image"]["line"]["width"]
        
        """ [draw.cam_image] """
        self.cam_weight: float = self.config["draw"]["cam_image"]["weight"]
        
        """ [draw.cam_image.replace_color] """
        self.replace_cam_color: bool = self.config["draw"]["cam_image"]["replace_color"]["enable"]
        self.replaced_colormap: int = getattr(cv2, self.config["draw"]["cam_image"]["replace_color"]["colormap"])
        
        """ [draw.cam_image.text] """
        self.text_font_style: Union[str, list] = self.config["draw"]["cam_image"]["text"]["font_style"]
        self.text_font_size: Union[int, list] = self.config["draw"]["cam_image"]["text"]["font_size"]
        
        """ [draw.cam_image.text.color] """
        self.text_correct_color: list = self.config["draw"]["cam_image"]["text"]["color"]["correct"]
        self.text_incorrect_color: list = self.config["draw"]["cam_image"]["text"]["color"]["incorrect"]
        self.text_shadow_color: list = self.config["draw"]["cam_image"]["text"]["color"]["shadow"]
        # ---------------------------------------------------------------------/


    def _set_history_dir(self):
        """
        """
        if self.model_state not in ["best", "final"]:
            raise ValueError(f"(config) `model_prediction.state`: "
                             f"'{self.model_state}', accept 'best' or 'final' only\n")
        
        model_prediction: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("model_prediction")
        
        # assort dir
        best_found = []
        final_found = []
        found_list = list(model_prediction.glob(f"{self.model_time_stamp}*"))
        tmp_dict = {i: path for i, path in enumerate(found_list)}
        for i, path in enumerate(found_list):
            if f"{{best}}" in str(path): best_found.append(tmp_dict.pop(i))
            elif f"{{final}}" in str(path): final_found.append(tmp_dict.pop(i))
        found_list = list(tmp_dict.values())
        
        # best mark
        if self.model_state == "best" and best_found:
            assert_0_or_1_history_dir(best_found, self.model_time_stamp, self.model_state)
            self.history_dir = best_found[0]
            return
        
        # final mark
        if self.model_state == "final" and final_found:
            assert_0_or_1_history_dir(final_found, self.model_time_stamp, self.model_state)
            self.history_dir = final_found[0]
            return
        
        # unset ( original )
        assert_0_or_1_history_dir(found_list, self.model_time_stamp, self.model_state)
        if found_list:
            self.history_dir = found_list[0]
            return
        else:
            raise ValueError("No `history_dir` matches the provided config")
        # ---------------------------------------------------------------------/


    def _set_training_config_attrs(self):
        """
        """
        path = self.history_dir.joinpath("training_config.toml")
        if not path.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} "
                                    f"Can't find 'training_config.toml' "
                                    f"( loss the most important file ). "
                                    f"{Style.RESET_ALL}\n")
        
        self.training_config: Union[dict, TOMLDocument] = \
                                load_config(path, cli_out=self._cli_out)
        
        """ [dataset] """
        self.dataset_seed_dir: str = self.training_config["dataset"]["seed_dir"]
        self.dataset_data: str = self.training_config["dataset"]["data"]
        self.dataset_palmskin_result: str = self.training_config["dataset"]["palmskin_result"]
        self.dataset_classif_strategy: str = self.training_config["dataset"]["classif_strategy"]
        self.dataset_file_name: str = self.training_config["dataset"]["file_name"]
        # ---------------------------------------------------------------------/


    def _set_config_attrs_default_value(self):
        """
        """
        """ [layout] """
        if not self.column:
            crop_size = parse_dataset_file_name(self.dataset_file_name)["crop_size"]
            if crop_size == 512: self.column = 5
            elif crop_size == 256: self.column = 13
        
        """ [draw.drop_image.line] """
        if not self.line_color: self.line_color = (180, 160, 0)
        if not self.line_width: self.line_width = 2
        
        """ [draw.cam_image] """
        if not self.cam_weight: self.cam_weight = 0.5
        
        """ [draw.cam_image.text] """
        if not self.text_font_style: self.text_font_style = get_mono_font()
        if not self.text_font_size: self.text_font_size = None
        
        """ [draw.cam_image.text.color] """
        if not self.text_correct_color: self.text_correct_color = (0, 255, 0)
        if not self.text_incorrect_color: self.text_incorrect_color = (255, 255, 255)
        if not self.text_shadow_color: self.text_shadow_color = (0, 0, 0)
        # ---------------------------------------------------------------------/


    def _set_src_root(self):
        """
        """
        dataset_cropped: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
        
        self.src_root = dataset_cropped.joinpath(self.dataset_seed_dir,
                                                 self.dataset_data,
                                                 self.dataset_palmskin_result)
        # ---------------------------------------------------------------------/


    def _set_test_df(self):
        """
        """
        dataset_file: Path = self.src_root.joinpath(self.dataset_classif_strategy,
                                                    self.dataset_file_name)
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} "
                                    f"Can't find target dataset file "
                                    f"run `1.2.create_dataset_file.py` to create it. "
                                    f"{Style.RESET_ALL}\n")
        
        dataset_df: pd.DataFrame = \
            pd.read_csv(dataset_file, encoding='utf_8_sig')
        
        self.test_df: pd.DataFrame = \
            dataset_df[(dataset_df["dataset"] == "test")]
        # ---------------------------------------------------------------------/


    def _set_predict_ans_dict(self):
        """
        """
        log_path = self.history_dir.joinpath(r"{Logs}_PredByFish_predict_ans.log")
        if not log_path.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find file: "
                                    r"'{Logs}_PredByFish_predict_ans.log' "
                                    f"run `3.2.{{TestByFish}}_vit_b_16.py` to create it"
                                    f"{Style.RESET_ALL}\n")
        
        with open(log_path, 'r') as f_reader: 
            self.predict_ans_dict = json.load(f_reader)
        # ---------------------------------------------------------------------/


    def _set_cam_result_root(self):
        """
        """
        self.cam_result_root: Path = self.history_dir.joinpath("cam_result")
        if not self.cam_result_root.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} "
                                    f"Can't find directory: 'cam_result/' "
                                    f"run `3.2.{{TestByFish}}_vit_b_16.py` and "
                                    f"set (config) `cam.enable` = true. "
                                    f"{Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/


    def _set_cam_gallery_dir(self):
        """
        """
        self.cam_gallery_dir = self.history_dir.joinpath("+---CAM_Gallery")
        if self.cam_gallery_dir.exists():
            raise FileExistsError(f"{Fore.RED}{Back.BLACK} "
                                  f"Directory already exists: '{self.cam_gallery_dir}'. "
                                  f"To re-generate, please delete it manually. "
                                  f"{Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/


    def _set_rank_dict(self):
        """
        """
        self.rank_dict: dict = {}
        
        for i in range(10+1):
            if i < 5: self.rank_dict[i*10] = f"Match{str(i*10)}_(misMatch)"
            elif i == 10: self.rank_dict[i*10] = f"Match{str(i*10)}_(Full)"
            else: self.rank_dict[i*10] =  f"Match{str(i*10)}"
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """

        Args:
            config (Union[str, Path]): a toml file.
        """
        super().run(config)
        
        self._create_rank_dirs()
        
        fish_dsnames = sorted(Counter(self.test_df["parent (dsname)"]).keys(),
                              key=dsname.get_dsname_sortinfo)
        # fish_dsnames = fish_dsnames[:5] # for debug
        
        self._cli_out.divide()
        self._progressbar = tqdm(total=len(fish_dsnames), desc=f"[ {self._cli_out.logger_name} ] : ")
        
        for fish_dsname in fish_dsnames:
            self._progressbar.desc = f"[ {self._cli_out.logger_name} ] Generating '{fish_dsname}' "
            self._progressbar.refresh()
            self.gen_single_cam_gallery(fish_dsname)
        
        self._progressbar.close()
        self._del_empty_rank_dirs()
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _create_rank_dirs(self):
        """
        """
        num2class_list = sorted(Counter(self.test_df["class"]).keys())
        
        for key in num2class_list:
            for _, value in self.rank_dict.items():
                create_new_dir(self.cam_gallery_dir.joinpath(key, value))
        # ---------------------------------------------------------------------/


    def gen_single_cam_gallery(self, fish_dsname:str):
        """
        """
        fish_cls = self._get_fish_cls(fish_dsname)
        
        test_preserve_paths, \
            test_discard_paths, \
                cam_result_paths = self._get_path_lists(fish_dsname)
        
        self._read_images_as_dict(test_preserve_paths, # --> self.test_preserve_img_dict
                                  test_discard_paths,  # --> self.test_discard_img_dict
                                  cam_result_paths)    # --> self.cam_result_img_dict
        
        # >>> draw on 'discard' images <<<
        
        for name, bgr_img in self.test_discard_img_dict.items():
            self._draw_on_drop_image(name, bgr_img)
        
        # >>> draw on `cam` images <<<
        
        self.correct_cnt: int = 0
        for (cam_name, cam_img), (preserve_name, preserve_bgr_img) \
            in zip(self.cam_result_img_dict.items(), self.test_preserve_img_dict.items()):
            self._draw_on_cam_image(cam_name, cam_img, preserve_name, preserve_bgr_img)
        
        # >>> check which `rank_dir` to store <<<
        self._calculate_correct_rank()
        
        # >>> orig: `test_preserve_img_dict`, `test_discard_img_dict` <<<
        self._gen_orig_gallery(fish_dsname, fish_cls)
        
        # >>> overlay: `cam_result_img_dict`, `test_discard_img_dict` <<<
        self._gen_overlay_gallery(fish_dsname, fish_cls)
        
        # >>> update pbar <<<
        self._progressbar.update(1)
        self._progressbar.refresh()
        # ---------------------------------------------------------------------/


    def _get_fish_cls(self, fish_dsname:str) -> str:
        """
        """
        df = self.test_df[(self.test_df["parent (dsname)"] == fish_dsname)]
        
        cnt = Counter(df["class"])
        fish_cls = cnt.most_common(1)[0][0]
        
        return fish_cls
        # ---------------------------------------------------------------------/


    def _get_path_lists(self, fish_dsname:str)-> Tuple[list, list, list]:
        """
        """   
        # >>> cam result <<<
        
        cam_result_paths: list[Path] = []
        if self.replace_cam_color:
            cam_result_paths = sorted(self.cam_result_root.glob(f"{fish_dsname}/grayscale_map/*.tiff"),
                                          key=dsname.get_dsname_sortinfo)
        else:
            cam_result_paths = sorted(self.cam_result_root.glob(f"{fish_dsname}/color_map/*.tiff"),
                                          key=dsname.get_dsname_sortinfo)
        cam_dict: dict[int, Path] = \
            {dsname.get_dsname_sortinfo(path)[-1]: path for path in cam_result_paths}
        
        # >>> test_df <<<
        
        df = self.test_df[(self.test_df["parent (dsname)"] == fish_dsname)]
        tmp_dict: dict[int, Path] = \
            {dsname.get_dsname_sortinfo(path)[-1]: \
                self.src_root.joinpath(path) for path in df["path"]}
        
        # >>> Seperate 'predict' / 'not predict' ( without CAM ) <<<
        
        # predict (preserve)
        test_preserve_paths: list[Path] = []
        for num in cam_dict.keys():
            test_preserve_paths.append(tmp_dict.pop(num))
        
        # not predict (discard)
        test_discard_paths: list[Path] = list(tmp_dict.values())
        
        # >>> return <<<
        return test_preserve_paths, test_discard_paths, cam_result_paths
        # ---------------------------------------------------------------------/


    def _read_images_as_dict(self, test_preserve_paths:list, 
                                  test_discard_paths:list, 
                                  cam_result_paths:list):
        """
        """
        self.test_preserve_img_dict: dict[str, np.ndarray] = \
            { os.path.split(os.path.splitext(path)[0])[-1]: \
                cv2.imread(str(path)) for path in test_preserve_paths }
        
        self.test_discard_img_dict: dict[str, np.ndarray] = \
            { os.path.split(os.path.splitext(path)[0])[-1]: \
                cv2.imread(str(path)) for path in test_discard_paths }
        
        self.cam_result_img_dict: dict[str, np.ndarray] = \
            { os.path.split(os.path.splitext(path)[0])[-1]: \
                cv2.imread(str(path)) for path in cam_result_paths }
        # ---------------------------------------------------------------------/


    def _draw_on_drop_image(self, name:str, bgr_img:np.ndarray):
        """
        """
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = np.uint8(rgb_img * 0.5) # suppress brightness
        rgb_img = Image.fromarray(rgb_img) # convert to pillow image before drawing
        draw_x_on_image(rgb_img, self.line_color, self.line_width)
        self.test_discard_img_dict[name] = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        # ---------------------------------------------------------------------/


    def _draw_on_cam_image(self, cam_name:str, cam_img: np.ndarray,
                                preserve_name:str, preserve_bgr_img: np.ndarray):
        """
        """
        # >>> preparing `cam_img` <<<
        
        if self.replace_cam_color: cam_bgr_img = cv2.applyColorMap(cam_img, self.replaced_colormap) # BGR
        else: cam_bgr_img = cam_img
        cam_rgb_img = cv2.cvtColor(cam_bgr_img, cv2.COLOR_BGR2RGB)
        
        # >>> preparing `preserve_img` <<<
        preserve_rgb_img = cv2.cvtColor(preserve_bgr_img, cv2.COLOR_BGR2RGB)
        
        # >>> overlay `cam_img` on `preserve_img` <<<
        
        cam_overlay = ((cam_rgb_img/255) * self.cam_weight + 
                       (preserve_rgb_img/255) * (1 - self.cam_weight))
        cam_overlay = np.uint8(255 * cam_overlay)
        
        # >>> if the prediction is wrong, add answer on image <<<
        
        # get param for `draw_predict_ans_on_image`
        gt_cls = self.predict_ans_dict[preserve_name]['gt']
        pred_cls = self.predict_ans_dict[preserve_name]['pred']
        
        if gt_cls != pred_cls:
            # create a red mask
            mask = np.zeros_like(cam_overlay) # black mask
            mask[:, :, 0] = 1 # modify to `red` mask
            # fusion with red mask
            mask_overlay = np.uint8(255 *((cam_overlay/255) * 0.7 + mask * 0.3))
            # draw text
            mask_overlay = Image.fromarray(mask_overlay) # convert to pillow image before drawing
            draw_predict_ans_on_image(mask_overlay, pred_cls, gt_cls,
                                      self.text_font_style, self.text_font_size,
                                      self.text_correct_color,
                                      self.text_incorrect_color,
                                      self.text_shadow_color)
            cam_overlay = np.array(mask_overlay)
        else:
            self.correct_cnt += 1
            # show correct 'BG'
            if gt_cls == "BG":
                cam_overlay = Image.fromarray(cam_overlay)
                draw_predict_ans_on_image(cam_overlay, pred_cls, gt_cls,
                                          self.text_font_style, self.text_font_size,
                                          self.text_correct_color,
                                          self.text_incorrect_color,
                                          self.text_shadow_color)
                cam_overlay = np.array(cam_overlay)
        
        # >>> replace image <<<
        self.cam_result_img_dict[cam_name] = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)
        # ---------------------------------------------------------------------/


    def _calculate_correct_rank(self):
        """
        """
        self.matching_ratio_percent = int((self.correct_cnt / len(self.cam_result_img_dict))*100)
        for key, value in self.rank_dict.items():
            if self.matching_ratio_percent >= key: self.cls_matching_state = value
        # ---------------------------------------------------------------------/


    def _gen_orig_gallery(self, fish_dsname:str, fish_cls):
        """
        """
        orig_img_dict: dict = deepcopy(self.test_preserve_img_dict)
        orig_img_dict.update(self.test_discard_img_dict)
        sorted_orig_img_dict = OrderedDict(sorted(list(orig_img_dict.items()), key=lambda x: dsname.get_dsname_sortinfo(x[0])))
        orig_img_list = [ img for _, img in sorted_orig_img_dict.items() ]
        
        # >>> plot with 'Auto Row Calculation' <<<
        
        figtitle = (f"( original ) [{fish_cls}] {fish_dsname} : "
                    f"{self.dataset_palmskin_result}, "
                    f"{os.path.splitext(self.dataset_file_name)[0]}")
        save_path = self.cam_gallery_dir.joinpath(fish_cls, self.cls_matching_state, 
                                                  f"{fish_dsname}_orig.png")
        kwargs_plot_with_imglist_auto_row = {
            "img_list"   : orig_img_list,
            "column"     : self.column,
            "fig_dpi"    : 200,
            "figtitle"   : figtitle,
            "save_path"  : save_path,
            "show_fig"   : False
        }
        plot_with_imglist_auto_row(**kwargs_plot_with_imglist_auto_row)
        # ---------------------------------------------------------------------/


    def _gen_overlay_gallery(self, fish_dsname, fish_cls):
        """
        """
        cam_overlay_img_dict: dict = deepcopy(self.cam_result_img_dict)
        cam_overlay_img_dict.update(self.test_discard_img_dict)
        sorted_cam_overlay_img_dict = OrderedDict(sorted(list(cam_overlay_img_dict.items()), key=lambda x: dsname.get_dsname_sortinfo(x[0])))
        cam_overlay_img_list = [ img for _, img in sorted_cam_overlay_img_dict.items() ]
        
        # >>> plot with 'Auto Row Calculation' <<<
        
        figtitle = (f"( cam overlay ) [{fish_cls}] {fish_dsname} : "
                    f"{self.dataset_palmskin_result}, "
                    f"{os.path.splitext(self.dataset_file_name)[0]}, "
                    f"correct : {self.correct_cnt}/{len(self.cam_result_img_dict)} ({self.matching_ratio_percent/100})")
        save_path = self.cam_gallery_dir.joinpath(fish_cls, self.cls_matching_state, 
                                                  f"{fish_dsname}_overlay.png")
        kwargs_plot_with_imglist_auto_row = {
            "img_list"   : cam_overlay_img_list,
            "column"     : self.column,
            "fig_dpi"    : 200,
            "figtitle"   : figtitle,
            "save_path"  : save_path,
            "show_fig"   : False
        }
        plot_with_imglist_auto_row(**kwargs_plot_with_imglist_auto_row)
        # ---------------------------------------------------------------------/


    def _del_empty_rank_dirs(self):
        """
        """
        num2class_list = sorted(Counter(self.test_df["class"]).keys())
        
        for key in num2class_list:
            for _, value in self.rank_dict.items():
                rank_dir = self.cam_gallery_dir.joinpath(key, value)
                pngs = list(rank_dir.glob("**/*.png"))
                if len(pngs) == 0:
                    shutil.rmtree(rank_dir)
        # ---------------------------------------------------------------------/