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
import skimage as ski
from colorama import Back, Fore, Style
from PIL import Image
from rich.traceback import install
from tomlkit.toml_document import TOMLDocument
from tqdm.auto import tqdm

from ....data.dataset.dsname import get_dsname_sortinfo
from ....data.dataset.utils import parse_dataset_file_name
from ....dl.tester.utils import get_history_dir
from ....dl.utils import gen_class2num_dict
from ....shared.baseobject import BaseObject
from ....shared.config import load_config
from ....shared.utils import create_new_dir
from ...utils import (draw_drop_info_on_image, draw_predict_ans_on_image,
                      draw_x_on_image, get_font, plot_with_imglist_auto_row)
from .utils import get_gallery_column

install()
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
        self._set_dataset_param()
        self._set_config_attrs_default_value()
        
        self._set_src_root()
        self._set_test_df()
        self._set_mapping_attrs()
        self._set_predict_ans_dict()
        self._read_predbyfish_report()
        self._set_cam_result_root()
        
        self._set_cam_gallery_dir()
        # self._set_rank_dict()
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
        self.history_dir = get_history_dir(self._path_navigator,
                                           self.model_time_stamp,
                                           self.model_state,
                                           cli_out=self._cli_out)
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
        self.dataset_base_size: str = self.training_config["dataset"]["base_size"]
        self.dataset_classif_strategy: str = self.training_config["dataset"]["classif_strategy"]
        self.dataset_file_name: str = self.training_config["dataset"]["file_name"]
        
        """ [train_opts.data] """
        self.add_bg_class: bool = self.training_config["train_opts"]["data"]["add_bg_class"]
        # ---------------------------------------------------------------------/


    def _set_dataset_param(self) -> None:
        """
        """
        name: str = self.training_config["dataset"]["file_name"]
        self.dataset_param = parse_dataset_file_name(name)
        # ---------------------------------------------------------------------/


    def _set_config_attrs_default_value(self):
        """
        """
        """ [layout] """
        if not self.column:
            self.column = get_gallery_column(self.dataset_base_size,
                                                self.dataset_file_name)
        
        """ [draw.drop_image.line] """
        if not self.line_color: self.line_color = (180, 160, 0)
        if not self.line_width: self.line_width = 2
        
        """ [draw.cam_image] """
        if not self.cam_weight: self.cam_weight = 0.5
        
        """ [draw.cam_image.text] """
        if not self.text_font_style: self.text_font_style = \
                            str(get_font(alt_default_family="monospace"))
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
                                                 self.dataset_palmskin_result,
                                                 self.dataset_base_size)
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


    def _set_mapping_attrs(self):
        """ Set below attributes
            >>> self.num2class_list: list
            >>> self.class2num_dict: dict[str, int]
        
        Example :
        >>> num2class_list = ['L', 'M', 'S']
        >>> class2num_dict = {'L': 0, 'M': 1, 'S': 2}
        """
        cls_list = list(Counter(self.test_df["class"]).keys())
        
        if self.add_bg_class:
            cls_list.append("BG")
        
        self.num2class_list: list = sorted(cls_list)
        self.class2num_dict: dict[str, int] = gen_class2num_dict(self.num2class_list)
        
        self._cli_out.write(f"num2class_list = {self.num2class_list}, "
                            f"class2num_dict = {self.class2num_dict}")
        # ---------------------------------------------------------------------/


    def _set_predict_ans_dict(self):
        """
        """
        file_name = r"{Logs}_PredByFish_predict_ans.log"
        
        log_path = self.history_dir.joinpath(file_name)
        if not log_path.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find file: '{file_name}'"
                                    f"run proper script under `4.test_by_fish` to create it"
                                    f"{Style.RESET_ALL}\n")
        
        with open(log_path, 'r') as f_reader:
            self.predict_ans_dict = json.load(f_reader)
        # ---------------------------------------------------------------------/


    def _read_predbyfish_report(self):
        """
        """
        file_name = r"{Report}_PredByFish.log"
        
        log_path = self.history_dir.joinpath(file_name)
        if not log_path.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find file: '{file_name}'"
                                    f"run proper script under `4.test_by_fish/` to create it"
                                    f"{Style.RESET_ALL}\n")
        
        with open(log_path, 'r') as f_reader:
            self.predbyfish_report = f_reader.read()
        # ---------------------------------------------------------------------/


    def _set_cam_result_root(self):
        """
        """
        self.cam_result_root: Path = self.history_dir.joinpath("cam_result")
        if not self.cam_result_root.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} "
                                    f"Can't find directory: 'cam_result/' "
                                    f"run proper script under `4.test_by_fish/` and "
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


    def _set_rank_dict(self): # deprecated
        """ (deprecated)
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
        
        # self._create_rank_dirs()
        
        fish_dsnames = sorted(Counter(self.test_df["parent (dsname)"]).keys(),
                              key=get_dsname_sortinfo)
        # fish_dsnames = fish_dsnames[:5] # for debug
        
        self._cli_out.divide()
        self._progressbar = tqdm(total=len(fish_dsnames), desc=f"[ {self._cli_out.logger_name} ] : ")
        
        for fish_dsname in fish_dsnames:
            self._progressbar.desc = f"[ {self._cli_out.logger_name} ] Generating '{fish_dsname}' "
            self._progressbar.refresh()
            self.gen_single_cam_gallery(fish_dsname)
        
        self._progressbar.close()
        # self._del_empty_rank_dirs()
        self._cli_out.new_line()
        # ---------------------------------------------------------------------/


    def _create_rank_dirs(self): # deprecated
        """ (deprecated)
        """
        for key in self.num2class_list:
            for _, value in self.rank_dict.items():
                create_new_dir(self.cam_gallery_dir.joinpath(key, value))
        # ---------------------------------------------------------------------/


    def reset_single_fish_attrs(self):
        """
        """
        self.com_gt: str = ""
        self.com_pred: str = ""
        
        self.tested_img_dict: dict[str, np.ndarray] = {}
        self.untest_img_dict: dict[str, np.ndarray] = {}
        self.cam_result_img_dict: dict[str, np.ndarray] = {}
        
        self.correct_cnt: int = 0
        self.accuracy: float = 0.0
        self.content: str = ""
        # ---------------------------------------------------------------------/


    def gen_single_cam_gallery(self, fish_dsname:str):
        """
        """
        self.reset_single_fish_attrs()
        self.com_gt = self._get_com_cls(fish_dsname, "gt")
        self.com_pred = self._get_com_cls(fish_dsname, "pred")
        
        tested_paths, \
            untest_paths, \
                cam_result_paths = self._get_path_lists(fish_dsname)
        
        self._read_images_as_dict(tested_paths, # --> self.tested_img_dict
                                  untest_paths,  # --> self.untest_img_dict
                                  cam_result_paths)  # --> self.cam_result_img_dict
        
        # >>> draw on 'untest' images <<<
        for untest_name, untest_img in self.untest_img_dict.items():
            self._draw_on_drop_image(untest_name, untest_img)
        
        # >>> draw on `cam` images <<<
        for (cam_name, cam_img), (tested_name, tested_img) \
            in zip(self.cam_result_img_dict.items(), self.tested_img_dict.items()):
                self._draw_on_cam_image(cam_name, cam_img,
                                        tested_name, tested_img)
        
        # >>> preparing information which adds to the gallery <<<
        self.accuracy = self.correct_cnt / len(self.cam_result_img_dict)
        self.content = self._gen_detail_info_content(fish_dsname)
        
        # >>> orig: `tested_img_dict` + `untest_img_dict` <<<
        self._gen_orig_gallery(fish_dsname)
        
        # >>> overlay: `cam_result_img_dict` + `untest_img_dict` <<<
        self._gen_overlay_gallery(fish_dsname)
        
        # >>> update pbar <<<
        self._progressbar.update(1)
        self._progressbar.refresh()
        # ---------------------------------------------------------------------/


    def _get_com_cls(self, fish_dsname:str, key: str) -> str:
        """

        Args:
            fish_dsname (str): e.g. 'fish_1_A'
            key (str): 'gt' or 'pred'

        Returns:
            str: common class for sub-crops
        """
        if key not in ["gt", "pred"]:
            raise ValueError(f"param 'key', accept 'gt' or 'pred' only\n")
        
        df = self.test_df[(self.test_df["parent (dsname)"] == fish_dsname)]
        
        # get voted class
        class_cnt: Counter = Counter()
        for crop_name in df["image_name"]:
            try:
                # if image is tested
                class_cnt.update([self.predict_ans_dict[crop_name][key]])
            except KeyError:
                pass
        
        return class_cnt.most_common(1)[0][0]
        # ---------------------------------------------------------------------/


    def _get_path_lists(self, fish_dsname:str)-> tuple[list, list, list]:
        """
        """   
        # >>> cam result (tested) <<<
        
        cam_result_paths: list[Path] = []
        if self.replace_cam_color:
            cam_result_paths = sorted(self.cam_result_root.glob(f"{fish_dsname}/grayscale_map/*.tiff"),
                                          key=get_dsname_sortinfo)
        else:
            cam_result_paths = sorted(self.cam_result_root.glob(f"{fish_dsname}/color_map/*.tiff"),
                                          key=get_dsname_sortinfo)
        cam_dict: dict[int, Path] = \
            {get_dsname_sortinfo(path)[-1]: path for path in cam_result_paths}
        
        # >>> test_df <<<
        
        df = self.test_df[(self.test_df["parent (dsname)"] == fish_dsname)]
        tmp_dict: dict[int, Path] = \
            {get_dsname_sortinfo(path)[-1]: \
                self.src_root.joinpath(path) for path in df["path"]}
        
        # >>> Seperate 'tested' / 'untest' (without CAM) <<<
        
        # tested (predict)
        tested_paths: list[Path] = []
        for crop_sn in cam_dict.keys():
            tested_paths.append(tmp_dict.pop(crop_sn))
        
        # untest (not predict)
        untest_paths: list[Path] = list(tmp_dict.values())
        
        # >>> return <<<
        return tested_paths, untest_paths, cam_result_paths
        # ---------------------------------------------------------------------/


    def _read_images_as_dict(self, tested_paths:list,
                                   untest_paths:list,
                                   cam_result_paths:list):
        """
        """
        # self.tested_img_dict: dict[str, np.ndarray] = \
        #     { os.path.split(os.path.splitext(path)[0])[-1]: \
        #         cv2.imread(str(path)) for path in tested_paths }
        self.tested_img_dict: dict[str, np.ndarray] = \
            { path.stem: ski.io.imread(path) for path in tested_paths }
        
        # self.untest_img_dict: dict[str, np.ndarray] = \
        #     { os.path.split(os.path.splitext(path)[0])[-1]: \
        #         cv2.imread(str(path)) for path in untest_paths }
        self.untest_img_dict: dict[str, np.ndarray] = \
            { path.stem: ski.io.imread(path) for path in untest_paths }
        
        # self.cam_result_img_dict: dict[str, np.ndarray] = \
        #     { os.path.split(os.path.splitext(path)[0])[-1]: \
        #         cv2.imread(str(path)) for path in cam_result_paths }
        self.cam_result_img_dict: dict[str, np.ndarray] = \
            { path.stem: ski.io.imread(path) for path in cam_result_paths }
        # ---------------------------------------------------------------------/


    def _draw_on_drop_image(self, untest_name:str, untest_img:np.ndarray):
        """
        """
        assert untest_img.dtype == np.uint8, "untest_img.dtype != np.uint8"
        
        rgb_img = Image.fromarray(np.uint8(untest_img*0.5)) # convert to pillow image before drawing
        draw_x_on_image(rgb_img, self.line_color, self.line_width)
        self._add_dark_ratio_on_image(untest_name, rgb_img)
        
        # >>> replace image <<<
        assert id(untest_img) != id(rgb_img)
        self.untest_img_dict[untest_name] = np.array(rgb_img)
        # ---------------------------------------------------------------------/


    def _add_dark_ratio_on_image(self, img_name: str, rgb_image:Image.Image):
        """
        """
        target_row = self.test_df[(self.test_df["image_name"] == img_name)]
        assert len(target_row) == 1, f"Find {len(target_row)} '{img_name}'"
        
        dark_ratio = float(list(target_row["dark_ratio"])[0])
        draw_drop_info_on_image(rgb_image, self.dataset_param["intensity"],
                                dark_ratio, self.dataset_param["drop_ratio"])
        # ---------------------------------------------------------------------/


    def _draw_on_cam_image(self, cam_name:str, cam_img:np.ndarray,
                                 tested_name:str, tested_img:np.ndarray):
        """
        """
        assert get_dsname_sortinfo(cam_name) == get_dsname_sortinfo(tested_name)
        assert cam_img.dtype == np.uint8, "cam_img.dtype != np.uint8"
        assert tested_img.dtype == np.uint8, "tested_img.dtype != np.uint8"
        
        # preparing `cam_rgb_img` (np.float64)
        if self.replace_cam_color:
            cam_bgr_img = cv2.applyColorMap(cam_img, self.replaced_colormap) # BGR
            cam_rgb_img = cv2.cvtColor(cam_bgr_img, cv2.COLOR_BGR2RGB)/255.0
        else:
            cam_rgb_img = cam_img/255.0
        
        # preparing `tested_rgb_img` (np.float64)
        tested_rgb_img = tested_img/255.0
        
        # overlay `cam_img` on `tested_img`
        cam_overlay = (cam_rgb_img*self.cam_weight + 
                       tested_rgb_img*(1 - self.cam_weight))
        
        # get 'sub-crop' predicted results for `draw_predict_ans_on_image`
        gt_cls = self.predict_ans_dict[tested_name]['gt']
        pred_cls = self.predict_ans_dict[tested_name]['pred']
        
        if pred_cls != gt_cls:
            # create a red mask
            mask = np.zeros_like(cam_overlay, dtype=np.float64) # black mask
            mask[:, :, 0] = 1.0 # modify to `red` mask
            # fusion with red mask
            mask_overlay = cam_overlay*0.7 + mask*0.3
            mask_overlay = np.uint8(mask_overlay*255)
            # draw text
            rgb_img = Image.fromarray(mask_overlay) # convert to pillow image before drawing
            draw_predict_ans_on_image(rgb_img, pred_cls, gt_cls,
                                      self.text_font_style, self.text_font_size,
                                      self.text_correct_color,
                                      self.text_incorrect_color,
                                      self.text_shadow_color)
            cam_overlay = np.array(rgb_img)
        else:
            self.correct_cnt += 1
            cam_overlay = np.uint8(cam_overlay*255)
            # for `add_bg_class` flag
            if self.com_gt != gt_cls:
                rgb_img = Image.fromarray(cam_overlay) # convert to pillow image before drawing
                draw_predict_ans_on_image(rgb_img, pred_cls, gt_cls,
                                          self.text_font_style, self.text_font_size,
                                          self.text_correct_color,
                                          self.text_incorrect_color,
                                          self.text_shadow_color)
                cam_overlay = np.array(rgb_img)
        
        # >>> replace image <<<
        assert id(cam_img) != id(cam_overlay)
        self.cam_result_img_dict[cam_name] = cam_overlay
        # ---------------------------------------------------------------------/


    def _calculate_correct_rank(self): # deprecated
        """ (deprecated)
        """
        self.matching_ratio_percent = int((self.correct_cnt / len(self.cam_result_img_dict))*100)
        for key, value in self.rank_dict.items():
            if self.matching_ratio_percent >= key: self.cls_matching_state = value
        # ---------------------------------------------------------------------/


    def _gen_detail_info_content(self, fish_dsname:str):
        """
        """
        content = []
        
        # fish_dsname
        content.extend(["➣ ", f"image name   : {fish_dsname}", "\n"*2])
        
        # `{Logs}_PredByFish_predict_ans.log`
        content.extend(["➣ ", f'ground truth : "{self.com_gt}"', "\n"*1])
        if self.com_pred == self.com_gt:
            content.extend(["➣ ", f'predict      : "correct"', "\n"*2])
        else:
            content.extend(["➣ ", f'predict      : "{self.com_pred}"', "\n"*2])
        
        # accuracy
        content.extend(["➣ ", "accuracy     : "])
        content.extend([f"{self.correct_cnt}/{len(self.cam_result_img_dict)} "])
        content.extend([f"({self.accuracy:.5f})", "\n"*1])
        
        # avg. predicted probability
        content.extend(["➣ ", "avg. predicted probability : "])
        content.extend([json.dumps(self._cal_avg_pred_prob(fish_dsname)), "\n"*2])
        
        # `training_config.toml`
        content.extend(["➣ ", "training_config.note :", "\n"*1])
        content.extend([self.training_config["note"], "\n"*2])
        
        # `{Report}_PredByFish.log`
        content.extend([self.predbyfish_report])
        content = "".join(content)
        
        # adjust line height
        content = content.replace("\n", "@")
        content = content.replace("@", "\n"*2)
        
        return content
        # ---------------------------------------------------------------------/


    def _cal_avg_pred_prob(self, fish_dsname:str) -> dict[str, float]:
        """

        Args:
            fish_dsname (str): e.g. 'fish_1_A'

        Returns:
            str: common class for sub-crops
        """
        df = self.test_df[(self.test_df["parent (dsname)"] == fish_dsname)]
        
        # get voted class
        avg_pred_prob: dict[str, list] = {"L": [], "M": [], "S": []}
        
        for crop_name in df["image_name"]:
            pred_prob: dict[str, float] = self.predict_ans_dict[crop_name]["pred_prob"]
            for k, prob in pred_prob.items():
                avg_pred_prob[k].append(prob)
        
        for k, probs in avg_pred_prob.items():
            avg_pred_prob[k] = round(np.average(probs), 5)
        
        return avg_pred_prob
        # ---------------------------------------------------------------------/


    def _gen_orig_gallery(self, fish_dsname:str):
        """
        """
        img_dict: dict = deepcopy(self.tested_img_dict)
        img_dict.update(self.untest_img_dict)
        sorted_img_dict = OrderedDict(sorted(list(img_dict.items()), key=lambda x: get_dsname_sortinfo(x[0])))
        
        # add `dark_ratio` on images
        for crop_name, img in sorted_img_dict.items():
            rgb_img = Image.fromarray(img)
            self._add_dark_ratio_on_image(crop_name, rgb_img)
            sorted_img_dict[crop_name] = np.array(rgb_img)

        # >>> plot with 'Auto Row Calculation' <<<
        
        img_list = [ img for _, img in sorted_img_dict.items() ]
        
        subtitle_list = [ " " for _ in sorted_img_dict.keys() ]
        
        rel_path = f"{self.com_gt}/{self.accuracy:0.5f}_{fish_dsname}_orig.png"
        save_path = self.cam_gallery_dir.joinpath(rel_path)
        create_new_dir(save_path.parent)
        
        kwargs_plot_with_imglist_auto_row = {
            "img_list"      : img_list,
            "column"        : self.column,
            "fig_dpi"       : 200,
            "content"       : self.content,
            "subtitle_list" : subtitle_list,
            "save_path"     : save_path,
            "use_rgb"       : True,
            "show_fig"      : False
        }
        plot_with_imglist_auto_row(**kwargs_plot_with_imglist_auto_row)
        # ---------------------------------------------------------------------/


    def _gen_overlay_gallery(self, fish_dsname:str):
        """
        """
        img_dict: dict = deepcopy(self.cam_result_img_dict)
        img_dict.update(self.untest_img_dict)
        sorted_img_dict = OrderedDict(sorted(list(img_dict.items()), key=lambda x: get_dsname_sortinfo(x[0])))
        
        # >>> plot with 'Auto Row Calculation' <<<
        
        img_list = [ img for _, img in sorted_img_dict.items() ]
        
        subtitle_list = []
        for crop_name in sorted_img_dict.keys():
            if self.replace_cam_color: 
                crop_name = crop_name.replace("graymap", "crop")
            else: 
                crop_name = crop_name.replace("colormap", "crop")
            tmp_str = json.dumps(self.predict_ans_dict[crop_name]["pred_prob"])
            subtitle_list.append(tmp_str)
        
        rel_path = f"{self.com_gt}/{self.accuracy:0.5f}_{fish_dsname}_overlay.png"
        save_path = self.cam_gallery_dir.joinpath(rel_path)
        create_new_dir(save_path.parent)
        
        kwargs_plot_with_imglist_auto_row = {
            "img_list"      : img_list,
            "column"        : self.column,
            "fig_dpi"       : 200,
            "content"       : self.content,
            "subtitle_list" : subtitle_list,
            "save_path"     : save_path,
            "use_rgb"       : True,
            "show_fig"      : False
        }
        plot_with_imglist_auto_row(**kwargs_plot_with_imglist_auto_row)
        # ---------------------------------------------------------------------/


    def _del_empty_rank_dirs(self): # deprecated
        """ (deprecated)
        """
        for key in self.num2class_list:
            for _, value in self.rank_dict.items():
                rank_dir = self.cam_gallery_dir.joinpath(key, value)
                pngs = list(rank_dir.glob("**/*.png"))
                if len(pngs) == 0:
                    shutil.rmtree(rank_dir)
        # ---------------------------------------------------------------------/