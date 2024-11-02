# -*- coding: utf-8 -*-
"""
"""
import os
import pickle
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich.console import Console
from rich.pretty import Pretty
from rich.traceback import install
from scipy.ndimage import binary_dilation as dila
from skimage.segmentation import mark_boundaries, slic

from ..shared.config import load_config
from ..shared.utils import create_new_dir
from .calc_seg_feat import count_element, update_seg_analysis_dict
from .utils import get_seg_desc, get_slic_param_name

install()
# -----------------------------------------------------------------------------/


def bwRGB(bw,im):
    """
    """
    A = np.sum(bw)
    B = np.sum(im[bw,0])/A
    G = np.sum(im[bw,1])/A
    R = np.sum(im[bw,2])/A
    return [R,G,B]
    # -------------------------------------------------------------------------/


def col_dis(color1,color2):
    """
    """
    sum = 0
    for i in range(3):
        ds = (float(color1[i]) - float(color2[i]))**2
        sum =sum+ds
    delta_e = np.sqrt(sum)
    return delta_e
    # -------------------------------------------------------------------------/


def save_segment_result(save_path:Path, seg:np.ndarray):
    """
    """
    with open(save_path, mode="wb") as f:
        pickle.dump(seg, f)
    # -------------------------------------------------------------------------/


def save_seg_on_img(save_path:Path, img:np.ndarray, seg:np.ndarray):
    """
    """
    seg_on_img = np.uint8(mark_boundaries(img, seg)*255)
    cv2.imwrite(str(save_path), seg_on_img)
    # -------------------------------------------------------------------------/


def single_slic_labeling(dir:Path, img_path:Path,
                         n_segments:int, dark:int, merge:int,
                         debug_mode:bool=False):
    """
    """
    result_name = img_path.stem.split(".")[0]
    slic_param_name = f"S{n_segments}_D{dark}_M{merge}"
    img_name = f"{result_name}.{slic_param_name}"
    
    """ read image """
    img = cv2.imread(str(img_path))
    
    """ run SLIC """
    seg0 = slic(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                n_segments = n_segments,
                channel_axis=-1,
                convert2lab=True,
                enforce_connectivity=True,
                slic_zero=False, compactness=30,
                max_num_iter=100,
                sigma = [1.7,1.7],
                spacing=[1,1], # 3D: z, y, x; 2D: y, x
                min_size_factor=0.4,
                max_size_factor=3,
                start_label=0)
        # parameters can refer to https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic

    """ save original `seg_result` ( without merge ) """
    save_path = dir.joinpath(f"{img_name}.seg0.pkl")
    save_segment_result(save_path, seg0)

    """ overlapping original image with its `seg_result` """
    save_path = dir.joinpath(f"{img_name}.seg0.png")
    save_seg_on_img(save_path, img, seg0)

    """ merging neighbors ('black background' and 'similar color') """
    labels = np.unique(seg0)
    max_label = np.max(labels)
    lindex = max_label + 1 # new (re-index) labels on seg1 starts from `max_label` + 1
    seg1 = deepcopy(seg0) # copy the slic segment to 're-index' and 'merge'
    for label in labels:
        bw = (seg1 == label)
        
        if debug_mode:
            tmp_array = np.zeros_like(bw, dtype=np.uint8)
            tmp_array[bw] = 255  # 将True值设置为255，以渲染为白色
            plt.imshow(tmp_array, cmap='gray', vmax=255, vmin=0)
            plt.title(f"label = {label}")
            plt.show()
        
        if np.sum(bw) > 0: # SLIC 生成的 labels 會跳號，bw 可能會沒東西
            color1 = bwRGB(bw, img)
            color_dist = col_dis(color1, [0, 0, 0]) # compare with 'black background'
            if color_dist <= dark:
                seg1[(seg1 == label)] = 0 # dark region on seg1 is labeled as 0
            else:
                seg1[(seg1 == label)] = lindex # re-index
                lindex +=1
        else:
            print(f"'{label}' has been merged before dealing with")

    """ save merged `seg_result` """
    save_path = dir.joinpath(f"{img_name}.seg1.pkl")
    save_segment_result(save_path, seg1)

    """ overlapping original image with merged `seg_result` """
    save_path = dir.joinpath(f"{img_name}.seg1.png")
    save_seg_on_img(save_path, img, seg1)
    
    # >>> Merge similar color <<<
    
    seg2 = deepcopy(seg1)
    labels = np.unique(seg2)
    for label in labels:
        if label != 0:
            bw = (seg2 == label)
            if np.sum(bw) > 0: # merge 後會跳號，bw 可能會沒東西
                color1 = bwRGB(bw, img) # get self color
                bwd = dila(bw) # touch neighbor
                nlabels = np.unique(seg2[bwd]) # self + neighbor's labels
                for nl in nlabels:
                    if (nl > label) and (nl != 0):
                        bw2 = (seg2 == nl)
                        color2 = bwRGB(bw2, img) # neighbor's color
                        if col_dis(color1, color2) <= merge:
                            seg2[bw2] = label
            else:
                if debug_mode:
                    print(f"'{label}' has been merged before dealing with")
    
    """ save merged `seg_result` """
    save_path = dir.joinpath(f"{img_name}.seg2.pkl")
    save_segment_result(save_path, seg2)

    """ overlapping original image with merged `seg_result` """
    save_path = dir.joinpath(f"{img_name}.seg2.png")
    save_seg_on_img(save_path, img, seg2)
    
    return seg1, seg2
    # -------------------------------------------------------------------------/


def single_cellpose_prediction():
    """ Function name TBD
    Place holder for running Cellpose prediction
    """
    pass
    # -------------------------------------------------------------------------/


if __name__ == '__main__':

    """ Init components """
    console = Console()
    console.print(f"\nPy Module: '{Path(__file__)}'\n")
    
    # colloct image file names
    img_dir = Path(r"") # directory of input images, images extension: .tif / .tiff

    # scan TIFF files
    img_paths = list(img_dir.glob("*.tif*"))
    console.print(f"Found {len(img_paths)} TIFF files.")
    console.rule()

    """ Load config """
    config_name: str = "ml_analysis.toml"
    config = load_config(config_name)
    # [SLIC]
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    merge: int       = config["SLIC"]["merge"]
    debug_mode: bool = config["SLIC"]["debug_mode"]
    # [seg_results]
    seg_desc = get_seg_desc(config)
    console.print(f"Config : '{config_name}'\n",
                    Pretty(config, expand_all=True))
    console.rule()
    
    for img_path in img_paths:
        
        # get `seg_dirname`
        if seg_desc == "SLIC":
            seg_param_name = get_slic_param_name(config)
        elif seg_desc == "Cellpose":
            seg_param_name = "model_id" # TBD
        seg_dirname = f"{img_path.stem}.{seg_param_name}"
        
        seg_dir = img_path.parent.joinpath(f"{seg_desc}/{seg_dirname}")
        create_new_dir(seg_dir)
        
        # generate cell segmentation
        if seg_desc == "SLIC":
            seg1, seg2 = single_slic_labeling(seg_dir, img_path,
                                              n_segments, dark, merge,
                                              debug_mode)
        elif seg_desc == "Cellpose":
            seg1, seg2 = single_cellpose_prediction()
        # Note : `seg1` is 1st merge (background), `seg2` is 2nd merge (color)
        
        # save cell segmentation feature
        analysis_dict = {}
        analysis_dict = update_seg_analysis_dict(analysis_dict, *count_element(seg1, "cell"))
        save_path = seg_dir.joinpath(f"cell_count_{analysis_dict['cell_count']}")
        with open(save_path, mode="w") as f_writer: pass # empty file
        
        console.print(f"'{seg_desc}' of '{img_path.name}' save to : '{save_path.parent}'")
    
    console.line()
    console.print("[green]Done! \n")
    # -------------------------------------------------------------------------/