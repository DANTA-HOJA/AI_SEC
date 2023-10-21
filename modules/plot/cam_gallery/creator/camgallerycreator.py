import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from collections import Counter, OrderedDict
from copy import deepcopy
import json

import cv2
from PIL import Image
import matplotlib; matplotlib.use("agg")
import numpy as np
import pandas as pd
from tomlkit.toml_document import TOMLDocument
from colorama import Fore, Back, Style
from tqdm.auto import tqdm

from ...utils import draw_x_on_image, draw_predict_ans_on_image, \
                     plot_with_imglist_auto_row, get_mono_font
from ....data.dataset import dsname
from ....data.dataset.utils import parse_dataset_xlsx_name
from ....shared.clioutput import CLIOutput
from ....shared.config import load_config
from ....shared.pathnavigator import PathNavigator
from ....shared.utils import create_new_dir
from ....assert_fn import assert_0_or_1_history_dir
# -----------------------------------------------------------------------------/


class CamGalleryCreator:


    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        self._path_navigator = PathNavigator()
        self._cli_out = CLIOutput(display_on_CLI, 
                                  logger_name="Cam Gallery Creator")
        
        # ---------------------------------------------------------------------
        # """ attributes """
        # TODO
        # ---------------------------------------------------------------------/



    def _set_attrs(self, config_file:Union[str, Path]):
        """
        """
        self.config: Union[dict, TOMLDocument] = load_config(config_file, cli_out=self._cli_out)
        self._set_config_attrs()
        self._set_history_dir()
        self._set_train_config_attrs()
        self._set_dataset_xlsx_path()
        
        """ Load `dataset_xlsx` """
        self.dataset_xlsx_df: pd.DataFrame = pd.read_excel(self.dataset_xlsx_path, engine='openpyxl')
        
        self._set_attrs_default_value()
        self._set_cam_result_root()
        self._set_cam_gallery_dir()
        self._set_predict_ans_dict()
        self._set_rank_dict()
        # ---------------------------------------------------------------------/



    def run(self, config_file:Union[str, Path]="4.make_cam_gallery.toml"):
        """
        """
        self._cli_out.divide()
        self._set_attrs(config_file)
        
        fish_dsname_list = [ str(path).split(os.sep)[-1] for path in list(self.cam_result_root.glob("*")) ]
        fish_dsname_list = sorted(fish_dsname_list, key=dsname.get_dsname_sortinfo)
        
        self.create_rank_dirs()
        
        self._cli_out.divide()
        self.progressbar = tqdm(total=len(fish_dsname_list), desc=f"[ {self._cli_out.logger_name} ] : ")
            
        for fish_dsname in fish_dsname_list:
            self.progressbar.desc = f"[ {self._cli_out.logger_name} ] Generating '{fish_dsname}' "
            self.progressbar.refresh()
            self.gen_single_cam_gallery(fish_dsname)
        
        self.progressbar.close()
        self._cli_out.new_line()
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
            raise ValueError(f"config: `model_prediction.state`: "
                             f"'{self.model_state}', accept 'best' or 'final' only\n")
        
        model_prediction: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("model_prediction")
        
        best_found = []
        final_found = []
        found_list = list(model_prediction.glob(f"{self.model_time_stamp}*"))
        for i, path in enumerate(found_list):
            if f"{{best}}" in str(path): best_found.append(found_list.pop(i))
            if f"{{final}}" in str(path): final_found.append(found_list.pop(i))

        if self.model_state == "best" and best_found:
            assert_0_or_1_history_dir(best_found, self.model_time_stamp, self.model_state)
            self.history_dir = best_found[0]
            return
        
        if self.model_state == "final" and final_found:
            assert_0_or_1_history_dir(final_found, self.model_time_stamp, self.model_state)
            self.history_dir = final_found[0]
            return
        
        assert_0_or_1_history_dir(found_list, self.model_time_stamp, self.model_state)
        if found_list:
            self.history_dir = found_list[0]
            return
        else:
            raise ValueError("No `history_dir` matches the provided config")
        # ---------------------------------------------------------------------/



    def _set_train_config_attrs(self):
        """
        """
        path = self.history_dir.joinpath("train_config.toml")
        if not path.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find 'train_config.toml' "
                                    f"( loss the most important file ). "
                                    f"{Style.RESET_ALL}\n")
        
        self.train_config: Union[dict, TOMLDocument] = load_config(path, cli_out=self._cli_out)
        
        """ [dataset] """
        self.dataset_seed_dir: str = self.train_config["dataset"]["seed_dir"]
        self.dataset_name: str = self.train_config["dataset"]["name"]
        self.dataset_result_alias: str = self.train_config["dataset"]["result_alias"]
        self.dataset_classif_strategy: str = self.train_config["dataset"]["classif_strategy"]
        self.dataset_xlsx_name: str = self.train_config["dataset"]["xlsx_name"]
        # ---------------------------------------------------------------------/



    def _set_dataset_xlsx_path(self):
        """
        """
        dataset_cropped: Path = \
            self._path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v2")
        
        self.dataset_xlsx_path: Path = dataset_cropped.joinpath(self.dataset_seed_dir,
                                                                self.dataset_name,
                                                                self.dataset_result_alias,
                                                                self.dataset_classif_strategy,
                                                                f"{self.dataset_xlsx_name}.xlsx")
        if not self.dataset_xlsx_path.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find `dataset_xlsx` "
                                    f"run `1.3.create_dataset_xlsx.py` to create it. "
                                    f"{Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/



    def _set_attrs_default_value(self):
        """
        """
        """ [layout] """
        if not self.column:
            crop_size = parse_dataset_xlsx_name(self.dataset_xlsx_name)["crop_size"]
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



    def _set_cam_result_root(self):
        """
        """
        self.cam_result_root = self.history_dir.joinpath("cam_result")
        if not self.cam_result_root.exists():
            raise FileNotFoundError(f"{Fore.RED}{Back.BLACK} Can't find directory: 'cam_result/' "
                                    f"run `3.2.{{TestByFish}}_vit_b_16.py` and set `cam.enable` = True. "
                                    f"{Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/



    def _set_cam_gallery_dir(self):
        """
        """
        self.cam_gallery_dir = self.history_dir.joinpath("!--- CAM Gallery")
        if self.cam_gallery_dir.exists():
            raise FileExistsError(f"{Fore.RED}{Back.BLACK} Directory already exists: '{self.cam_gallery_dir}'. "
                                  f"To re-generate, please delete it manually. "
                                  f"{Style.RESET_ALL}\n")
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



    def _set_rank_dict(self):
        """
        """
        self.rank_dict: dict = {}
        
        for i in range(10+1):
            if i < 5: self.rank_dict[i*10] = f"Match{str(i*10)}_(misMatch)"
            elif i == 10: self.rank_dict[i*10] = f"Match{str(i*10)}_(Full)"
            else: self.rank_dict[i*10] =  f"Match{str(i*10)}"
        # ---------------------------------------------------------------------/



    def create_rank_dirs(self):
        """
        """
        num2class_list = sorted(Counter(self.dataset_xlsx_df["class"]).keys())
        
        for key in num2class_list:
            for _, value in self.rank_dict.items():
                create_new_dir(self.cam_gallery_dir.joinpath(key, value))
        # ---------------------------------------------------------------------/



    def gen_single_cam_gallery(self, fish_dsname:str):
        """
        """
        fish_cls = self.get_fish_cls(fish_dsname)
        
        test_preserve_path_list, \
            test_discard_path_list, \
                cam_result_path_list = self.get_path_lists(fish_dsname)
        
        self.read_images_as_dict(test_preserve_path_list, # --> self.test_preserve_img_dict
                                 test_discard_path_list,  # --> self.test_discard_img_dict
                                 cam_result_path_list)    # --> self.cam_result_img_dict
        
        # draw on 'discard' images
        for path, bgr_img in self.test_discard_img_dict.items():
            self.draw_on_drop_image(path, bgr_img)
        
        # draw on `cam` images
        self.pred_cls_cnt = Counter()
        for (cam_path, cam_img), (preserve_path, preserve_bgr_img) in zip(self.cam_result_img_dict.items(), self.test_preserve_img_dict.items()):
            self.draw_on_cam_image(cam_path, cam_img, preserve_path, preserve_bgr_img)
        
        # check which `rank_dir` to store
        self.calculate_correct_rank(fish_cls)
        
        # orig: `test_preserve_img_dict`, `test_discard_img_dict`
        self.gen_orig_gallery(fish_dsname, fish_cls)
        
        # overlay: `cam_result_img_dict`, `test_discard_img_dict`
        self.gen_overlay_gallery(fish_dsname, fish_cls)
        
        # update pbar
        self.progressbar.update(1)
        self.progressbar.refresh()
        # ---------------------------------------------------------------------/



    def get_fish_cls(self, fish_dsname:str) -> str:
        """
        """
        df_filtered_rows = \
            self.dataset_xlsx_df[(self.dataset_xlsx_df["parent (dsname)"] == fish_dsname)]
        
        class_cnt_dict = Counter(df_filtered_rows["class"])
        assert len(class_cnt_dict) == 1
        fish_cls = list(class_cnt_dict.keys())[0]
        
        return fish_cls
        # ---------------------------------------------------------------------/



    def get_path_lists(self, fish_dsname:str)-> Tuple[list, list, list]:
        """
        """        
        # test preserve
        df_filtered_rows = \
            self.dataset_xlsx_df[(self.dataset_xlsx_df["parent (dsname)"] == fish_dsname) &
                                 (self.dataset_xlsx_df["state"] == "preserve")]
        test_preserve_path_list = sorted(df_filtered_rows["path"], key=dsname.get_dsname_sortinfo)
        
        # test discard
        df_filtered_rows = \
            self.dataset_xlsx_df[(self.dataset_xlsx_df["parent (dsname)"] == fish_dsname) &
                                 (self.dataset_xlsx_df["state"] == "discard")]
        test_discard_path_list = sorted(df_filtered_rows["path"], key=dsname.get_dsname_sortinfo)
        
        # cam result
        if self.replace_cam_color:
            cam_result_path_list = sorted(self.cam_result_root.glob(f"{fish_dsname}/grayscale_map/*.tiff"),
                                          key=dsname.get_dsname_sortinfo)
        else:
            cam_result_path_list = sorted(self.cam_result_root.glob(f"{fish_dsname}/color_map/*.tiff"),
                                          key=dsname.get_dsname_sortinfo)
        
        return test_preserve_path_list, test_discard_path_list, cam_result_path_list
        # ---------------------------------------------------------------------/



    def read_images_as_dict(self, test_preserve_path_list:list, 
                                  test_discard_path_list:list, 
                                  cam_result_path_list:list):
        """
        """
        self.test_preserve_img_dict = { str(img_path): cv2.imread(str(img_path)) \
                                            for img_path in test_preserve_path_list }
        
        self.test_discard_img_dict = { str(img_path): cv2.imread(str(img_path)) \
                                            for img_path in test_discard_path_list }
        
        self.cam_result_img_dict = { str(img_path): cv2.imread(str(img_path)) \
                                            for img_path in cam_result_path_list }
        # ---------------------------------------------------------------------/



    def draw_on_drop_image(self, path:str, bgr_img:cv2.Mat):
        """
        """
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = np.uint8(rgb_img * 0.5) # suppress brightness
        rgb_img = Image.fromarray(rgb_img) # convert to pillow image before drawing
        draw_x_on_image(rgb_img, self.line_color, self.line_width)
        self.test_discard_img_dict[path] = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        # ---------------------------------------------------------------------/



    def draw_on_cam_image(self, cam_path:str, cam_img: cv2.Mat,
                                preserve_path:str, preserve_bgr_img: cv2.Mat):
        """
        """
        # preparing `cam_img`
        if self.replace_cam_color: cam_bgr_img = cv2.applyColorMap(cam_img, self.replaced_colormap) # BGR
        else: cam_bgr_img = cam_img
        cam_rgb_img = cv2.cvtColor(cam_bgr_img, cv2.COLOR_BGR2RGB)
        
        # preparing `selected_img`
        preserve_rgb_img = cv2.cvtColor(preserve_bgr_img, cv2.COLOR_BGR2RGB)
                
        # overlay `cam_img` on `selected_img`
        cam_overlay = ((cam_rgb_img/255) * self.cam_weight + 
                       (preserve_rgb_img/255) * (1 - self.cam_weight))
        cam_overlay = np.uint8(255 * cam_overlay)
            
        # get param for `draw_predict_ans_on_image`
        preserve_image_name = preserve_path.split(os.sep)[-1].split(".")[0]
        gt_cls = self.predict_ans_dict[preserve_image_name]['gt']
        pred_cls = self.predict_ans_dict[preserve_image_name]['pred']
        self.pred_cls_cnt.update(pred_cls)
         
        if gt_cls != pred_cls:
            # create a red mask
            mask = np.zeros_like(cam_overlay) # black mask
            mask[:, :, 0] = 1 # modify to `red` mask
            mask_overlay = np.uint8(255 *((cam_overlay/255) * 0.7 + mask * 0.3)) # fusion with red mask
            # draw text
            rgb_img = Image.fromarray(mask_overlay) # convert to pillow image before drawing
            draw_predict_ans_on_image(rgb_img, pred_cls, gt_cls,
                                      self.text_font_style, self.text_font_size,
                                      self.text_correct_color,
                                      self.text_incorrect_color,
                                      self.text_shadow_color)
            cam_overlay = np.array(rgb_img)
        
        self.cam_result_img_dict[cam_path] = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)
        # ---------------------------------------------------------------------/



    def calculate_correct_rank(self, fish_cls:str):
        """
        """
        self.matching_ratio_percent = int((self.pred_cls_cnt[fish_cls] / len(self.cam_result_img_dict))*100)
        for key, value in self.rank_dict.items():
            if self.matching_ratio_percent >= key: self.cls_matching_state = value
        # ---------------------------------------------------------------------/



    def gen_orig_gallery(self, fish_dsname:str, fish_cls):
        """
        """
        orig_img_dict: dict = deepcopy(self.test_preserve_img_dict)
        orig_img_dict.update(self.test_discard_img_dict)
        sorted_orig_img_dict = OrderedDict(sorted(list(orig_img_dict.items()), key=lambda x: dsname.get_dsname_sortinfo(x[0])))
        orig_img_list = [ img for _, img in sorted_orig_img_dict.items() ]
        
        # plot with 'Auto Row Calculation'
        kwargs_plot_with_imglist_auto_row = {
            "img_list"   : orig_img_list,
            "column"     : self.column,
            "fig_dpi"    : 200,
            "figtitle"   : f"( original ) {fish_dsname} : {orig_img_list[-1].shape[:2]}",
            "save_path"  : self.cam_gallery_dir.joinpath(fish_cls, self.cls_matching_state, f"{fish_dsname}_orig.png"),
            "show_fig"   : False
        }
        plot_with_imglist_auto_row(**kwargs_plot_with_imglist_auto_row)
        # ---------------------------------------------------------------------/



    def gen_overlay_gallery(self, fish_dsname, fish_cls):
        """
        """
        cam_overlay_img_dict: dict = deepcopy(self.cam_result_img_dict)
        cam_overlay_img_dict.update(self.test_discard_img_dict)
        sorted_cam_overlay_img_dict = OrderedDict(sorted(list(cam_overlay_img_dict.items()), key=lambda x: dsname.get_dsname_sortinfo(x[0])))
        cam_overlay_img_list = [ img for _, img in sorted_cam_overlay_img_dict.items() ]
        
        # plot with 'Auto Row Calculation'
        kwargs_plot_with_imglist_auto_row = {
            "img_list"   : cam_overlay_img_list,
            "column"     : self.column,
            "fig_dpi"    : 200,
            "figtitle"   : (f"( cam overlay ) {fish_dsname} : {cam_overlay_img_list[-1].shape[:2]}, "
                            f"correct : {self.pred_cls_cnt[fish_cls]}/{len(self.cam_result_img_dict)} ({self.matching_ratio_percent/100})") ,
            "save_path"  : self.cam_gallery_dir.joinpath(fish_cls, self.cls_matching_state, f"{fish_dsname}_overlay.png"),
            "show_fig"   : False
        }
        plot_with_imglist_auto_row(**kwargs_plot_with_imglist_auto_row)
        # ---------------------------------------------------------------------/