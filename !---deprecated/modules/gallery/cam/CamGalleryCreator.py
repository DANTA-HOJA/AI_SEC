import os
import sys
import re
from typing import List, Dict, Tuple
from collections import Counter, OrderedDict
from pathlib import Path
from copy import deepcopy
import json
import toml

from tqdm.auto import tqdm
import numpy as np
import cv2
from PIL import Image

from fileop import create_new_dir
from gallery.utils import draw_x_on_image, draw_predict_ans_on_image
from dataset.utils import sortFishNameForDataset
import plt_show

# print("="*100, "\n")



class CamGalleryCreator():
    
    def __init__(self, config_path:Path) -> None:
        
        # Load `make_cam_gallery.toml` -------------------------------------------------------------------------------------
        if isinstance(config_path, Path):
            with open(config_path, mode="r") as f_reader:
                self.config = toml.load(f_reader)
        else: raise TypeError("'config_path' should be a 'Path' object, please using `from pathlib import Path`")
        
        self.column = self.config["layout"]["column"]
        
        self.line_color = self.config["draw"]["drop_image"]["line"]["color"]
        self.line_width = self.config["draw"]["drop_image"]["line"]["width"]

        self.cam_weight = self.config["draw"]["cam_image"]["weight"]
        self.replace_cam_color = self.config["draw"]["cam_image"]["replace_color"]["enable"]
        self.replaced_colormap  = getattr(cv2, self.config["draw"]["cam_image"]["replace_color"]["colormap"])
        
        self.text_font_style      = self.config["draw"]["cam_image"]["text"]["font_style"]
        self.text_font_size       = self.config["draw"]["cam_image"]["text"]["font_size"] # if None, do auto-detection
        self.text_correct_color   = self.config["draw"]["cam_image"]["text"]["color"]["correct"]
        self.text_incorrect_color = self.config["draw"]["cam_image"]["text"]["color"]["incorrect"]
        self.text_shadow_color    = self.config["draw"]["cam_image"]["text"]["color"]["shadow"]

        self.load_dir_root = Path(self.config["model"]["history_root"])
        self.model_name    = self.config["model"]["model_name"]
        self.model_history = self.config["model"]["history"]
        
        # Load `train_config.toml` -------------------------------------------------------------------------------------
        self.load_dir = self.load_dir_root.joinpath(self.model_name, self.model_history)
        self.train_config_path = self.load_dir.joinpath("train_config.toml")

        with open(self.train_config_path, mode="r") as f_reader:
            self.train_config = toml.load(f_reader)

        self.dataset_root        = Path(self.train_config["dataset"]["root"])
        self.dataset_name             = self.train_config["dataset"]["name"]
        self.dataset_gen_method       = self.train_config["dataset"]["gen_method"]
        self.dataset_classif_strategy = self.train_config["dataset"]["classif_strategy"]
        self.dataset_param_name       = self.train_config["dataset"]["param_name"]

        # Generate `path_vars` -------------------------------------------------------------------------------------
        self.dataset_dir       = self.dataset_root.joinpath(self.dataset_name, self.dataset_gen_method, self.dataset_classif_strategy, self.dataset_param_name)
        self.test_selected_dir = self.dataset_dir.joinpath("test", "selected")
        self.test_drop_dir     = self.dataset_dir.joinpath("test", "drop")

        self.cam_result_root = self.load_dir.joinpath("cam_result")
        self.cam_gallery_dir = self.load_dir.joinpath("!--- CAM Gallery")

        # Other "Pre-define" Vars -------------------------------------------------------------------------------------
        self.class_counts_dict: Dict[str, int] = None
        self.predict_ans_dict: Dict[str, Dict[str, str]] = None
        self.rank_dict: Dict[int, str] = {}
        
        self.progressbar = None
        
        self.test_selected_path_list: List = None
        self.test_drop_path_list: List = None
        self.cam_result_path_list: List = None
        
        self.test_selected_img_dict: Dict[str, cv2.Mat] = None
        self.test_drop_img_dict: Dict[str, cv2.Mat] = None
        self.cam_result_img_dict: Dict[str, cv2.Mat] = None
        
        self.pred_cls_cnt: Counter = None
        self.cls_matching_state = None
        self.matching_ratio_percent = None

        # Run Composite actions -------------------------------------------------------------------------------------
        
        # End of `__init__` -------------------------------------------------------------------------------------
    
    
    def run(self):
    
        fish_dsname_list = [ str(path).split(os.sep)[-1] for path in list(self.cam_result_root.glob("*")) ]
        fish_dsname_list.sort()
        
        assert not os.path.exists(self.cam_gallery_dir), f"dir: '{self.cam_gallery_dir}' already exists"
        self.read_required_log()
        self.create_required_dir()
        
        self.progressbar = tqdm(total=len(fish_dsname_list), desc="CAM Gallery ")
        
        for fish_dsname in fish_dsname_list:
            self.gen_single_cam_gallery(fish_dsname)

        self.progressbar.close()
    
    
    def gen_single_cam_gallery(self, fish_dsname:str):
        
        self.progressbar.desc = f"Generate ' {fish_dsname} ' "
        self.progressbar.refresh()
        
        
        fish_dsname_split = re.split(" |_|-", fish_dsname)
        fish_cls = fish_dsname_split[0]
        self.pred_cls_cnt = Counter()
        
        
        self.scan_fish_img_path(fish_dsname, fish_cls) # update var:
                                                       #    self.test_selected_path_list 
                                                       #    self.test_drop_path_list
                                                       #    self.cam_result_path_list
        
        self.read_images_as_dict() # update var:
                                   #    self.test_selected_img_dict
                                   #    self.test_drop_img_dict
                                   #    self.cam_result_img_dict

        
        # draw on 'drop' images
        for path, bgr_img in self.test_drop_img_dict.items():
            self.draw_on_drop_image(path, bgr_img)
        
        
        # draw on `cam` images
        for (cam_path, cam_img), (selected_path, selected_bgr_img) in zip(self.cam_result_img_dict.items(), self.test_selected_img_dict.items()):
            self.draw_on_cam_image(cam_path, cam_img, selected_path, selected_bgr_img)
        
        
        self.calculate_correct_rank(fish_cls)
        
        # orig: `test_selected_img_dict`, `test_drop_img_dict`
        self.gen_orig_gallery(fish_dsname, fish_cls)
        
        # overlay: `cam_result_img_dict`, `test_drop_img_dict`
        self.gen_overlay_gallery(fish_dsname, fish_cls)
        
        
        self.progressbar.update(1)
        self.progressbar.refresh()  
    
    
    def read_required_log(self):
        
        logs_path = self.dataset_dir.joinpath(r"{Logs}_train_selected_summary.log")
        with open(logs_path, 'r') as f_reader: self.class_counts_dict = json.load(f_reader)
        
        logs_path = self.load_dir.joinpath(r"{Logs}_PredByFish_predict_ans.log")
        with open(logs_path, 'r') as f_reader: self.predict_ans_dict = json.load(f_reader)
    
    
    def create_required_dir(self): 
        """
            create `dir_name` with ranking
            
            required_var: `self.class_counts_dict`
        """
        for i in range(10+1):
            if i < 5: self.rank_dict[i*10] = f"Match{str(i*10)}_(misMatch)"
            elif i == 10: self.rank_dict[i*10] = f"Match{str(i*10)}_(Full)"
            else: self.rank_dict[i*10] =  f"Match{str(i*10)}"
        
        for key, _ in self.class_counts_dict.items():
            for _, value in self.rank_dict.items():
                create_new_dir(os.path.join(self.cam_gallery_dir, key, value), display_in_CLI=False)
    
    
    def scan_fish_img_path(self, fish_dsname, fish_cls):
        
        self.test_selected_path_list = sorted(list(self.test_selected_dir.glob(f"{fish_cls}/{fish_dsname}_selected_*.tiff")),
                                              key=sortFishNameForDataset)
        
        self.test_drop_path_list = sorted(list(self.test_drop_dir.glob(f"{fish_cls}/{fish_dsname}_drop_*.tiff")),
                                          key=sortFishNameForDataset)

        if self.replace_cam_color:
            self.cam_result_path_list = sorted(list(self.cam_result_root.glob(f"{fish_dsname}/grayscale_map/*.tiff")),
                                               key=sortFishNameForDataset)
        else:
            self.cam_result_path_list = sorted(list(self.cam_result_root.glob(f"{fish_dsname}/color_map/*.tiff")),
                                               key=sortFishNameForDataset)

    
    def read_images_as_dict(self):
        self.test_selected_img_dict = { str(img_path): cv2.imread(str(img_path)) for img_path in self.test_selected_path_list }
        self.test_drop_img_dict     = { str(img_path): cv2.imread(str(img_path)) for img_path in self.test_drop_path_list }
        self.cam_result_img_dict    = { str(img_path): cv2.imread(str(img_path)) for img_path in self.cam_result_path_list }

    
    def draw_on_drop_image(self, path, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = np.uint8(rgb_img * 0.5) # suppress brightness
        rgb_img = Image.fromarray(rgb_img) # convert to pillow image before drawing
        draw_x_on_image(rgb_img, self.line_color, self.line_width)
        self.test_drop_img_dict[path] = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        

    def draw_on_cam_image(self, cam_path, cam_img, selected_path, selected_bgr_img):
        
        # preparing `cam_img`
        if self.replace_cam_color: cam_bgr_img = cv2.applyColorMap(cam_img, self.replaced_colormap) # BGR
        else: cam_bgr_img = cam_img
        cam_rgb_img = cv2.cvtColor(cam_bgr_img, cv2.COLOR_BGR2RGB)
        
        # preparing `selected_img`
        selected_rgb_img = cv2.cvtColor(selected_bgr_img, cv2.COLOR_BGR2RGB)
                
        # overlay `cam_img` on `selected_img`
        cam_overlay = ((cam_rgb_img/255) * self.cam_weight + 
                       (selected_rgb_img/255) * (1 - self.cam_weight))
        cam_overlay = np.uint8(255 * cam_overlay)
            
        # get param for `draw_predict_ans_on_image`
        selected_image_name = selected_path.split(os.sep)[-1].split(".")[0]
        gt_cls = self.predict_ans_dict[selected_image_name]['gt']
        pred_cls = self.predict_ans_dict[selected_image_name]['pred']
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


    def calculate_correct_rank(self, fish_cls):
        
        self.matching_ratio_percent = int((self.pred_cls_cnt[fish_cls] / len(self.cam_result_path_list))*100)
        for key, value in self.rank_dict.items():
            if self.matching_ratio_percent >= key: self.cls_matching_state = value


    def gen_orig_gallery(self, fish_dsname:str, fish_cls):
    
        orig_img_dict = deepcopy(self.test_selected_img_dict)
        orig_img_dict.update(self.test_drop_img_dict)
        sorted_orig_img_dict = OrderedDict(sorted(list(orig_img_dict.items()), key=lambda x: sortFishNameForDataset(x[0])))
        orig_img_list = [ img for _, img in sorted_orig_img_dict.items() ]
        
        # plot with 'Auto Row Calculation'
        kwargs_plot_with_imglist_auto_row = {
            "img_list"   : orig_img_list,
            "column"     : self.column,
            "fig_dpi"    : 200,
            "figtitle"   : f"( original ) {fish_dsname} : {orig_img_list[-1].shape[:2]}",
            "save_path"  : f"{self.cam_gallery_dir}/{fish_cls}/{self.cls_matching_state}/{fish_dsname}_orig.png",
            "show_fig"   : False
        }
        plt_show.plot_with_imglist_auto_row(**kwargs_plot_with_imglist_auto_row)
    
    
    def gen_overlay_gallery(self, fish_dsname, fish_cls):
        
        cam_overlay_img_dict = deepcopy(self.cam_result_img_dict)
        cam_overlay_img_dict.update(self.test_drop_img_dict)
        sorted_cam_overlay_img_dict = OrderedDict(sorted(list(cam_overlay_img_dict.items()), key=lambda x: sortFishNameForDataset(x[0])))
        cam_overlay_img_list = [ img for _, img in sorted_cam_overlay_img_dict.items() ]
        
        # plot with 'Auto Row Calculation'
        kwargs_plot_with_imglist_auto_row = {
            "img_list"   : cam_overlay_img_list,
            "column"     : self.column,
            "fig_dpi"    : 200,
            "figtitle"   : (f"( cam overlay ) {fish_dsname} : {cam_overlay_img_list[-1].shape[:2]}, "
                            f"correct : {self.pred_cls_cnt[fish_cls]}/{len(self.cam_result_path_list)} ({self.matching_ratio_percent/100})") ,
            "save_path"  : f"{self.cam_gallery_dir}/{fish_cls}/{self.cls_matching_state}/{fish_dsname}_overlay.png",
            "show_fig"   : False
        }
        plt_show.plot_with_imglist_auto_row(**kwargs_plot_with_imglist_auto_row)