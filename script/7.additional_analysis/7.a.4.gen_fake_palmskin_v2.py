import colorsys
import os
import pickle as pkl
import random
import re
import sys
from pathlib import Path

import numpy as np
import skimage as ski
from matplotlib import colors as mcolors
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
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, get_repo_root

install()
# -----------------------------------------------------------------------------/


def gen_unique_random_color_pool(n_labels:list,
                                 existing_color_pool:dict=None,
                                 hsv_vthreshold:int=None) -> dict:
    """_summary_

    Args:
        n_labels (list): How many colors should generate
        existing_color_pool (dict, optional): If given, will extend the given color pool. Defaults to None.
        hsv_vthreshold (int, optional): Only collect the colors over this threshold. Defaults to None.

    Returns:
        dict: {Hex rgb: float rgb}
    """
    if existing_color_pool is None:
        color_pool = dict() # set 遇到重複的 value 不會 update
    else:
        color_pool = existing_color_pool
    
    color_pool["#000000"] = mcolors.hex2color("#000000")
    
    while len(color_pool) <= len(n_labels):
        
        # Generating a random number in between 0 and 2^24
        color = random.randrange(0, 2**24)
        hex_rgb = f"#{hex(color)[2:]:>06}" # `hex(color)` starts with `0x`
        float_rgb = mcolors.hex2color(hex_rgb) # range: [0.0, 1.0]
        
        if hsv_vthreshold is None:
            color_pool[f"{hex_rgb}"] = float_rgb # save color
        else:
            hsv_color = colorsys.rgb_to_hsv(*float_rgb)
            # 以 V channel (亮度) 進行判斷
            v_threshold = hsv_vthreshold/255 # 255 is `V_MAX` in OpenCV HSV color model
            if hsv_color[2] > v_threshold:
                color_pool[f"{hex_rgb}"] = float_rgb # save color
    
    color_pool.pop("#000000") # remove `black` (background)

    return color_pool
    # -------------------------------------------------------------------------/


def gen_colorless_palmskin(seg:np.ndarray, cytosol_gray: float=0.5,
                           border_color: tuple[float, float, float]=(1.0, 1.0, 1.0)) -> np.ndarray:
    """
    """
    fake_palmskin = np.full_like(seg,
                                 int(cytosol_gray*np.iinfo(np.uint8).max),
                                 dtype=np.uint8)
    
    fake_palmskin = mark_boundaries(fake_palmskin, seg, color=border_color)
    
    return fake_palmskin
    # -------------------------------------------------------------------------/


def gen_cellless_palmskin(seg:np.ndarray,
                          cytosol_gray: float=0.5) -> np.ndarray:
    """
    """
    fake_palmskin = np.zeros_like(seg, dtype=np.float64)
    fake_palmskin[seg > 0] = cytosol_gray
    
    return fake_palmskin
    # -------------------------------------------------------------------------/


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
    dataset_base_size: str = config["dataset"]["base_size"]
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
    palmskin_processed_dname_dirs = processed_di.palmskin_processed_dname_dirs_dict
    
    """ Apply SLIC on each image """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(paths))
        
        for path, dpath in zip(paths, palmskin_processed_dname_dirs.values()):
            
            # check dsname, dname is match
            assert get_dsname_sortinfo(path) == get_dname_sortinfo(dpath), "dsname, dname not match"
            
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
            unique_labels = np.unique(seg1)[1:] # 排除 `background` (0)
            
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
            fake_imgs["random_color2b"] = mark_boundaries(colored_seg1, seg1, color=(1.0, 1.0, 1.0))
            
            """Random color #1: load `clone_seg` (seg2, with clonal information)"""
            seg2_pkl = dname_dir.joinpath(f"SLIC/{result_name}_{{dark_{dark}}}/{result_name}.seg2.pkl")
            print(f"[ {seg2_pkl.parts[-1]} : '{seg2_pkl}' ]")
            with open(seg2_pkl, mode="rb") as f_reader:
                seg2 = pkl.load(f_reader)
            # 創建與 seg2 相同大小的 RGB image
            colored_seg2 = np.zeros((*seg2.shape, 3), dtype=np.float64) # 3 表示 RGB 三個通道
            # 確認所有唯一的 label
            unique_labels = np.unique(seg2)[1:] # 排除 `background` (0), 應少於 seg1
            
            # 參照 seg1 的 "label 顏色映射" 進行上顏色
            for label in unique_labels:
                colored_seg2[seg2 == label] = color_pool[label2hexrgb[label]]
            fake_imgs["random_color1"] = colored_seg2
            # draw border
            fake_imgs["random_color1b"] = mark_boundaries(colored_seg2, seg1, color=(1.0, 1.0, 1.0))
            
            """Color-less: palmskin (border white, cytosol 50%gray, assume no shading and background)"""
            fake_imgs["color_less"] = \
                gen_colorless_palmskin(seg1, cytosol_gray=0.5, border_color=(1.0, 1.0, 1.0))
            
            """Cell-less: palmskin (background and shading are `black`)"""
            fake_imgs["cell_less"] = gen_cellless_palmskin(seg1, cytosol_gray=0.5)

            """Save images"""
            fakeimg_dir = dpath.joinpath(f"FakeImage_v2/{dataset_palmskin_result}_{{dark_{dark}}}")
            create_new_dir(fakeimg_dir)
            for key, img in fake_imgs.items():
                save_path = fakeimg_dir.joinpath(f"{dataset_palmskin_result}.{key}.tif")
                assert img.dtype == np.float64 # make sure image is present in float format
                ski.io.imsave(save_path, np.uint8(img*255)) # float to 8-bit image
                print(f"{key} : '{save_path}'")
            cli_out.new_line()
            
            # update pbar
            pbar.advance(task)
    
    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/