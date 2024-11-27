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

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.dataset.dsname import get_dsname_sortinfo
from modules.data.dname import get_dname_sortinfo
from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.fakepalmskin.utils import (gen_singlecolor_palmskin,
                                           gen_unique_random_color_pool)
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, get_repo_root

install()
# -----------------------------------------------------------------------------/


if __name__ == '__main__':
    
    print(f"Repository: '{get_repo_root()}'")
    
    """ Init components """
    processed_di = ProcessedDataInstance()
    path_navigator = PathNavigator()
    cli_out = CLIOutput()
    cli_out.divide()

    # load config
    # `dark` and `merge` are two parameters as color space distance, determined by experiences
    config = load_config("get_cell_feature.toml")
    # [dataset]
    dataset_seed_dir: str = config["dataset"]["seed_dir"]
    dataset_data: str = config["dataset"]["data"]
    dataset_palmskin_result: str = config["dataset"]["palmskin_result"]
    dataset_base_size: str = "W512_H1024"
    # [SLIC]
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    merge: int       = config["SLIC"]["merge"]
    debug_mode: bool = config["SLIC"]["debug_mode"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()

    # set `random seed`
    random.seed(int(dataset_seed_dir.replace("RND", "")))
    
    """ Colloct image file names """
    dataset_cropped: Path = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
    src_root = dataset_cropped.joinpath(dataset_seed_dir,
                                        dataset_data,
                                        dataset_palmskin_result,
                                        dataset_base_size)
    paths = sorted(src_root.glob("*/*/*.tiff"), key=get_dsname_sortinfo)
    print(f"Total files: {len(paths)}")
    
    """ Processed Data Instance """
    instance_desc = re.split("{|}", dataset_data)[1]
    temp_dict = {"data_processed": {"instance_desc": instance_desc}}
    processed_di.parse_config(temp_dict)
    csv_path = processed_di.instance_root.joinpath("data.csv")
    df: pd.DataFrame = pd.read_csv(csv_path, encoding='utf_8_sig')
    palmskin_dnames = sorted(pd.concat([df["Palmskin Anterior (SP8)"],
                                        df["Palmskin Posterior (SP8)"]]),
                            key=get_dname_sortinfo)
    
    """ Apply SLIC on each image """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(paths))
        
        for path, dname in zip(paths, palmskin_dnames):
            
            # check dsname, dname is match
            assert get_dsname_sortinfo(path) == get_dname_sortinfo(dname), "dsname, dname not match"
            
            result_name = path.stem
            dname_dir = path.parents[0]
            
            # check analyze condiction is same
            verify_cfg = dname_dir.joinpath(f"SLIC/{result_name}_{{dark_{dark}}}/{{copy}}_get_cell_feature.toml")
            assert config == load_config(verify_cfg), f"`verify_cfg` not match, '{verify_cfg}'"
            
            fake_imgs = {} # 4 types

            """Random color #2: load `cell_seg` (seg1, without clonal information)"""
            seg1_pkl = dname_dir.joinpath(f"SLIC/{result_name}_{{dark_{dark}}}/{result_name}.seg1.pkl")
            print(f"[ {seg1_pkl.parts[-1]} : '{seg1_pkl}' ]")
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
            seg2_pkl = dname_dir.joinpath(f"SLIC/{result_name}_{{dark_{dark}}}/{result_name}.seg2.pkl")
            print(f"[ {seg2_pkl.parts[-1]} : '{seg2_pkl}' ]")
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
            dpath = processed_di.palmskin_processed_dname_dirs_dict[dname]
            fakeimg_dir = dpath.joinpath(f"FakeImage_v2/{dataset_palmskin_result}_{{dark_{dark}}}")
            create_new_dir(fakeimg_dir)
            for key, img in fake_imgs.items():
                save_path = fakeimg_dir.joinpath(f"{dataset_palmskin_result}.{key}.tif")
                assert img.dtype == np.float64 # make sure image pixels are present in float
                ski.io.imsave(save_path, np.uint8(img*255)) # float to 8-bit image
                print(f"{key} : '{save_path}'")
            cli_out.new_line()
            
            # update pbar
            pbar.advance(task)
    
    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/