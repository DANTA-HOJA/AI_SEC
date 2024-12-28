import os
import pickle as pkl
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skimage as ski
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install
from skimage.segmentation import mark_boundaries

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.fakepalmskin.utils import (gen_singlecolor_palmskin,
                                           gen_unique_random_color_pool)
from modules.ml.utils import (get_cellpose_param_name, get_seg_desc,
                              get_slic_param_name)
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, get_repo_root

install()
# -----------------------------------------------------------------------------/


if __name__ == '__main__':
    
    print(f"Repository: '{get_repo_root()}'")
    
    """ Init components """
    cli_out = CLIOutput()
    cli_out.divide()
    path_navigator = PathNavigator()
    processed_di = ProcessedDataInstance()
    processed_di.parse_config("ml_analysis.toml")

    """ Load config """
    config = load_config("ml_analysis.toml")
    # [data_processed]
    palmskin_result_name: Path = Path(config["data_processed"]["palmskin_result_name"])
    cluster_desc: str = config["data_processed"]["cluster_desc"]
    # [seg_results]
    seg_desc = get_seg_desc(config)
    # [Cellpose]
    cp_model_name: str = config["Cellpose"]["cp_model_name"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()

    # get `seg_dirname`
    if seg_desc == "SLIC":
        seg_param_name = get_slic_param_name(config)
    elif seg_desc == "Cellpose":
        # check model
        cp_model_dir = path_navigator.dbpp.get_one_of_dbpp_roots("model_cellpose")
        cp_model_path = cp_model_dir.joinpath(cp_model_name)
        if cp_model_path.is_file():
            seg_param_name = get_cellpose_param_name(config)
        else:
            raise FileNotFoundError(f"'{cp_model_path}' is not a file or does not exist")
    seg_dirname = f"{palmskin_result_name.stem}.{seg_param_name}"

    # set `random seed`
    random.seed(42)

    """ Apply SLIC on each image """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...",
                             total=len(processed_di.palmskin_processed_dname_dirs_dict))
        
        for dname_dir in processed_di.palmskin_processed_dname_dirs_dict.values():
            
            fake_imgs = {} # 4 types
            src_dir = dname_dir.joinpath(seg_desc, seg_dirname)

            """Random color #2: load `cell_seg` (seg1, without clonal information)"""
            seg1_pkl = src_dir.joinpath(f"{seg_dirname}.seg1.pkl")
            print(f"[ {dname_dir.parts[-1]} : '{seg1_pkl}' ]")
            with open(seg1_pkl, mode="rb") as f_reader:
                seg1 = pkl.load(f_reader)
            # 創建與 seg1 相同大小的 RGB image
            colored_seg1 = np.zeros((*seg1.shape, 3), dtype=np.float64) # 3 表示 RGB 三個通道
            # 確認所有唯一的 label
            unique_labels = np.unique(seg1)
            if 0 in unique_labels:
                unique_labels = unique_labels[1:] # 排除 `background` (0)
            
            # 依照 `unique_labels` 數量生成不重複的顏色
            color_pool = gen_unique_random_color_pool(unique_labels)
            assert len(unique_labels) == len(color_pool)
            
            # 將每個 label 對應的顏色填入圖像
            label2hexrgb: dict[str, str] = dict() # 供 seg2 查找
            for label, (hex_rgb, float_rgb) in zip(unique_labels, color_pool.items()):
                assert label != 0
                colored_seg1[seg1 == label] = float_rgb
                label2hexrgb[label] = hex_rgb
            fake_imgs["random_color2"] = colored_seg1
            # draw border
            fake_imgs["random_color2b"] = \
                mark_boundaries(colored_seg1, seg1, color=(1.0, 1.0, 1.0))
            
            """Random color #1: load `clone_seg` (seg2, with clonal information)"""
            seg2_pkl = src_dir.joinpath(f"{seg_dirname}.seg2.pkl")
            print(f"[ {dname_dir.parts[-1]} : '{seg2_pkl}' ]")
            with open(seg2_pkl, mode="rb") as f_reader:
                seg2 = pkl.load(f_reader)
            # 創建與 seg2 相同大小的 RGB image
            colored_seg2 = np.zeros((*seg2.shape, 3), dtype=np.float64) # 3 表示 RGB 三個通道
            # 確認所有唯一的 label
            unique_labels = np.unique(seg2) # 應少於 `seg1`
            if 0 in unique_labels:
                unique_labels = unique_labels[1:] # 排除 `background` (0)
            
            # 參照 seg1 的 "label 顏色映射" 進行上色
            for label in unique_labels:
                colored_seg2[seg2 == label] = color_pool[label2hexrgb[label]]
            fake_imgs["random_color1"] = colored_seg2
            # draw border
            fake_imgs["random_color1b"] = \
                mark_boundaries(colored_seg2, seg1, color=(1.0, 1.0, 1.0))

            """Color-less: palmskin (background and shading are `black`, cytosol `50%gray`, with `white` border)"""
            fake_imgs["color_less"] = \
                gen_singlecolor_palmskin(seg1, cytosol_color=0.5,
                                         border_color=1.0, bg_color=0.0)
            # assume no background, shading, abbr(without background): wobg
            fake_imgs["color_less_wobg"] = \
                gen_singlecolor_palmskin(seg1, cytosol_color=0.5,
                                         border_color=1.0)
            
            """Cell-less: palmskin (background and shading are `black`, cytosol `50%gray`, without border)"""
            fake_imgs["cell_less"] = \
                gen_singlecolor_palmskin(seg1, cytosol_color=0.5,
                                         border_color=0.5, bg_color=0.0)
            
            """Save images"""
            fakeimg_dir = dname_dir.joinpath(f"FakeImage_v2", seg_desc, seg_dirname)
            create_new_dir(fakeimg_dir)
            for key, img in fake_imgs.items():
                save_path = fakeimg_dir.joinpath(f"{seg_dirname}.{key}.tif")
                assert img.dtype == np.float64 # make sure image pixels are present in float
                ski.io.imsave(save_path, np.uint8(img*255)) # float to 8-bit image
                print(f"{key} : '{save_path}'")
            cli_out.new_line()
            
            # update pbar
            pbar.advance(task)
    
    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/