# -*- coding: utf-8 -*-
"""
"""
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage as ski
from PIL import Image, ImageDraw, ImageFont
from rich import print
from rich.console import Console
from rich.pretty import Pretty
from rich.traceback import install
from scipy.ndimage import binary_dilation as dila
from skimage import measure
from skimage.color import deltaE_ciede94, rgb2lab
from skimage.segmentation import mark_boundaries, slic

from ..shared.config import load_config
from ..shared.pathnavigator import PathNavigator
from ..shared.utils import create_new_dir
from .calc_seg_feat import count_element, update_seg_analysis_dict
from .utils import get_cellpose_param_name, get_seg_desc, get_slic_param_name

install()
# -----------------------------------------------------------------------------/


def bwRGB(bw,im):
    """ (deprecated) channel order of `im` is `BGR` (default to `cv2.imread()`)
    """
    A = np.sum(bw)
    B = np.sum(im[bw,0])/A
    G = np.sum(im[bw,1])/A
    R = np.sum(im[bw,2])/A
    return [R,G,B]
    # -------------------------------------------------------------------------/


def simple_col_dis(color1, color2):
    """
    """
    sum = 0
    for i in range(3):
        ds = (float(color1[i]) - float(color2[i]))**2
        sum =sum+ds
    delta_e = np.sqrt(sum)
    return delta_e

    # return np.sqrt(np.sum((np.array(color1) - np.array(color2))**2))
    # -------------------------------------------------------------------------/


def save_segment_result(save_path:Path, seg:np.ndarray):
    """
    """
    with open(save_path, mode="wb") as f_writer:
        pickle.dump(seg, f_writer)
    # -------------------------------------------------------------------------/


def save_seg_on_img(save_path:Path, img:np.ndarray, seg:np.ndarray):
    """ (deprecated)
    """
    seg_on_img = np.uint8(mark_boundaries(img, seg)*255)
    cv2.imwrite(str(save_path), seg_on_img)
    # -------------------------------------------------------------------------/


def get_average_rgb(mask: np.ndarray, rgb_img: np.ndarray,
                    avg_ratio: float):
    """
    """
    assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    assert isinstance(avg_ratio, float)
    
    # vars
    ch_order = {"R":0, "G":1, "B":2}
    avg_rgb_dict = {"R":float, "G":float, "B":float}
    
    for k, v in ch_order.items():
        pixels = np.sort(rgb_img[mask, v])[::-1]
        area = len(pixels)
        # average with partial pixels
        pixel_avg = np.average(pixels[:int(area*avg_ratio)])
        # update `avg_rgb`
        avg_rgb_dict[k] = pixel_avg
    
    return avg_rgb_dict, np.array(list(avg_rgb_dict.values()))
    # -------------------------------------------------------------------------/


def get_average_rgb_v2(mask: np.ndarray, rgb_img: np.ndarray,
                       qL: float, qR: float):
    """
    """
    assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    assert isinstance(qL, float)
    assert isinstance(qR, float)
    
    # vars
    ch_order = {"R":0, "G":1, "B":2}
    avg_rgb_dict = {"R":float, "G":float, "B":float}
    
    for k, v in ch_order.items():
        pixels = np.sort(rgb_img[mask, v])[::-1]
        val_qL = np.quantile(pixels, qL)
        val_qR = np.quantile(pixels, qR)
        # average with custom quantile interval
        pixels = pixels[(pixels >= val_qL) & (pixels <= val_qR)]
        pixel_avg = np.average(pixels)
        # update `avg_rgb`
        avg_rgb_dict[k] = pixel_avg
    
    return avg_rgb_dict, np.array(list(avg_rgb_dict.values()))
    # -------------------------------------------------------------------------/


def average_rgb_coloring(seg: np.ndarray, rgb_img: np.ndarray):
    """ channel order of `img` is `RGB`
    """
    assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    
    # vars
    avgcolor_img = np.zeros((*seg.shape, 3), dtype=rgb_img.dtype)
    
    labels = np.unique(seg)
    for label in labels:
        if label == 0: continue # skip background
        mask = (seg == label)
        # _, avg_rgb = get_average_rgb(mask, rgb_img, avg_ratio=0.5)
        _, avg_rgb = get_average_rgb_v2(mask, rgb_img, qL=0.5, qR=0.9)
        avgcolor_img[mask] = np.uint8(avg_rgb)
    
    # check and return
    assert id(avgcolor_img) != id(rgb_img)
    return avgcolor_img
    # -------------------------------------------------------------------------/


def merge_similar_rgb(seg: np.ndarray, rgb_img: np.ndarray,
                      merge: float, debug_mode: bool):
    """
    """
    assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    
    # vars
    merge_seg = deepcopy(seg)
    relabeling: dict[int, int] = {}
    delta_e_dict: dict[str, float] = {} # for debugger
    
    labels = np.unique(merge_seg)
    for label in labels:
        if label == 0: continue # skip background
        mask = (merge_seg == label)
        if np.sum(mask) > 0: # merge 後會跳號， mask 可能會沒東西
            # color = get_average_rgb(mask, rgb_img, avg_ratio=0.5)[1] # get self color
            color = get_average_rgb_v2(mask, rgb_img, qL=0.5, qR=0.9)[1] # get self color
            mask_dila = dila(mask, iterations=2) # find neighbor
            nlabels = np.unique(merge_seg[mask_dila]) # self + neighbor's labels
            for nlabel in nlabels:
                if nlabel == 0: continue # skip background
                elif nlabel > label: # avoid repeated merging
                    nmask = (merge_seg == nlabel)
                    # ncolor = get_average_rgb(nmask, rgb_img, avg_ratio=0.5)[1] # neighbor's color
                    ncolor = get_average_rgb_v2(nmask, rgb_img, qL=0.5, qR=0.9)[1] # neighbor's color
                    delta_e = deltaE_ciede94(rgb2lab(color/255.0), rgb2lab(ncolor/255.0))
                    delta_e_dict[f"{label}_cmp_{nlabel}"] = delta_e # for debugger
                    if delta_e <= merge:
                        merge_seg[nmask] = label
                        relabeling[nlabel] = label
        else:
            if debug_mode:
                print(f"'{label}' has been merged before dealing with")
    
    # check and return
    assert id(merge_seg) != id(seg)
    return merge_seg, relabeling
    # -------------------------------------------------------------------------/


def draw_label_on_image(seg: np.ndarray, rgb_img: np.ndarray,
                        relabeling: dict[str, str] = {}):
    """
    """
    assert rgb_img.dtype == np.uint8, "rgb_img.dtype != np.uint8"
    
    # 計算每個 label 區域的屬性
    props = measure.regionprops(seg)
    
    # 使用 PIL 在影像上寫入文字
    pil_img = Image.fromarray(rgb_img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()  # 使用預設字體，也可以指定自定義字體
    
    # 對每個區域的屬性進行迭代，並在重心位置標上 label 值
    for prop in props:
        
        cY, cX = prop.centroid
        
        if prop.label in relabeling:
            label_text = str(relabeling[prop.label])
        else:
            label_text = str(prop.label)
        
        # 計算文字的大小
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 計算置中後的文字位置
        text_position = (cX - text_width / 2, cY - text_height / 2)
        
        # 畫陰影
        shadow_offset = 2 # pixels
        shadow_position = (text_position[0] + shadow_offset, text_position[1] + shadow_offset)
        draw.text(shadow_position, label_text, fill="#000000", font=font)
        
        # 畫主要文字
        if prop.label in relabeling:
            draw.text(text_position, label_text, fill="#FF0000", font=font)
        else:
            draw.text(text_position, label_text, fill="#FFFFFF", font=font)
    
    # check and return
    assert id(pil_img) != id(rgb_img)
    return np.array(pil_img)
    # -------------------------------------------------------------------------/


def single_slic_labeling(dst_dir:Path, img_path:Path,
                         n_segments:int, dark:int, merge:int,
                         debug_mode:bool=False):
    """
    """
    img_name = dst_dir.name
    # read image
    img = ski.io.imread(img_path)

    """ SLIC (seg0) """
    seg0 = slic(img,
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

    """ Save 'SLIC' result (seg0, without any merge) """
    # save segmentation as pkl file
    save_path = dst_dir.joinpath(f"{img_name}.seg0.pkl")
    save_segment_result(save_path, seg0)
    # Mark `seg0` on `img`
    seg0_on_img = np.uint8(mark_boundaries(img, seg0, color=(0, 1, 1))*255)
    save_path = dst_dir.joinpath(f"{img_name}.seg0.png")
    ski.io.imsave(save_path, seg0_on_img)

    """ Merge background (seg1) """
    seg1 = deepcopy(seg0)
    labels = np.unique(seg0)
    new_label = np.max(labels) + 1 # new (re-index) label start
    for label in labels:
        mask = (seg1 == label)
        if np.sum(mask) > 0: # SLIC 生成的 labels 會跳號， mask 可能會沒東西
            color = get_average_rgb(mask, img, avg_ratio=1.0)[1]
            color_dist = simple_col_dis(color, (0, 0, 0)) # compare with 'background'
            if color_dist <= dark:
                seg1[mask] = 0 # background on `seg1` is set to 0
            else:
                seg1[mask] = new_label # re-index
                new_label +=1
        else:
            print(f"'{label}' is missing")

    """ Save 'Merge background' result (seg1) """
    # save segmentation as pkl file
    save_path = dst_dir.joinpath(f"{img_name}.seg1.pkl")
    save_segment_result(save_path, seg1)
    # Generate average 'RGB' of `img` and mark `seg1` labels on it
    avg_rgb = average_rgb_coloring(seg1, img)
    for k, v in {"o": img, "a": avg_rgb}.items():
        seg1_on_img = np.uint8(mark_boundaries(v, seg1, color=(0, 1, 1))*255)
        save_path = dst_dir.joinpath(f"{img_name}.seg1{k}.png")
        ski.io.imsave(save_path, seg1_on_img)
        seg1_on_img = draw_label_on_image(seg1, seg1_on_img)
        save_path = dst_dir.joinpath(f"{img_name}.seg1{k}l.png")
        ski.io.imsave(save_path, seg1_on_img)

    """ Merge similar RGB (seg2) """
    seg2, relabeling = merge_similar_rgb(seg1, img,
                                         merge=merge, debug_mode=debug_mode)

    """ Save 'Merge similar RGB' result (seg2) """
    # save segmentation as pkl file
    save_path = dst_dir.joinpath(f"{img_name}.seg2.pkl")
    save_segment_result(save_path, seg2)
    # Mark `seg2` labels and merged regions on `img` and `avg_rgb`
    for k, v in {"o": img, "a": avg_rgb}.items():
        seg2_on_img = np.uint8(mark_boundaries(v, seg2, color=(0, 1, 1))*255)
        save_path = dst_dir.joinpath(f"{img_name}.seg2{k}.png")
        ski.io.imsave(save_path, seg2_on_img)
        seg2_on_img = draw_label_on_image(seg1, seg2_on_img, relabeling=relabeling)
        save_path = dst_dir.joinpath(f"{img_name}.seg2{k}l.png")
        ski.io.imsave(save_path, seg2_on_img)
    
    return seg1, seg2
    # -------------------------------------------------------------------------/


def single_cellpose_prediction(dst_dir: Path, img_path: Path,
                               channels: int, cp_model, merge: int,
                               debug_mode: bool=False):
    """ Function name TBD
    Place holder for running Cellpose prediction
    """
    from cellpose import io as cpio
    
    img_name = dst_dir.name
    # read image
    img = cpio.imread(img_path)
    
    # predict segments
    seg1, flow, style = cp_model.eval(img, channels=channels)
    
    """ Save 'Merge background' result (seg1) """
    # save segmentation as pkl file
    save_path = dst_dir.joinpath(f"{img_name}.seg1.pkl")
    save_segment_result(save_path, seg1)
    # Generate average 'RGB' of `img` and mark `seg1` labels on it
    avg_rgb = average_rgb_coloring(seg1, img)
    for k, v in {"o": img, "a": avg_rgb}.items():
        seg1_on_img = np.uint8(mark_boundaries(v, seg1, color=(0, 1, 1))*255)
        save_path = dst_dir.joinpath(f"{img_name}.seg1{k}.png")
        ski.io.imsave(save_path, seg1_on_img)
        seg1_on_img = draw_label_on_image(seg1, seg1_on_img)
        save_path = dst_dir.joinpath(f"{img_name}.seg1{k}l.png")
        ski.io.imsave(save_path, seg1_on_img)

    """ Merge similar RGB (seg2) """
    seg2, relabeling = merge_similar_rgb(seg1, img,
                                         merge=merge, debug_mode=debug_mode)

    """ Save 'Merge similar RGB' result (seg2) """
    # save segmentation as pkl file
    save_path = dst_dir.joinpath(f"{img_name}.seg2.pkl")
    save_segment_result(save_path, seg2)
    # Mark `seg2` labels and merged regions on `img` and `avg_rgb`
    for k, v in {"o": img, "a": avg_rgb}.items():
        seg2_on_img = np.uint8(mark_boundaries(v, seg2, color=(0, 1, 1))*255)
        save_path = dst_dir.joinpath(f"{img_name}.seg2{k}.png")
        ski.io.imsave(save_path, seg2_on_img)
        seg2_on_img = draw_label_on_image(seg1, seg2_on_img, relabeling=relabeling)
        save_path = dst_dir.joinpath(f"{img_name}.seg2{k}l.png")
        ski.io.imsave(save_path, seg2_on_img)
    
    return seg1, seg2
    # -------------------------------------------------------------------------/


if __name__ == '__main__':

    """ Init components """
    console = Console()
    console.print(f"\nPy Module: '{Path(__file__)}'\n")
    path_navigator = PathNavigator()
    
    # colloct image file names
    img_dir = Path(r"") # directory of input images, images extension: .tif / .tiff

    # scan TIFF files
    img_paths = list(img_dir.glob("*.tif*"))
    console.print(f"Found {len(img_paths)} TIFF files.")
    console.rule()

    """ Load config """
    config_name: str = "ml_analysis.toml"
    config = load_config(config_name)
    # [seg_results]
    seg_desc = get_seg_desc(config)
    console.print(f"Config : '{config_name}'\n",
                  Pretty(config, expand_all=True))
    console.rule()
    # [SLIC]
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    # [Cellpose]
    cp_model_name: str = config["Cellpose"]["cp_model_name"]
    channels: list     = config["Cellpose"]["channels"]
    
    # get `seg_dirname`
    merge: int       = config[f"{seg_desc}"]["merge"]
    debug_mode: bool = config[f"{seg_desc}"]["debug_mode"]
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
        # load model
        if "cellpose" in sys.executable: # check python environment
                from cellpose import models as cpmodels
                cp_model = cpmodels.CellposeModel(gpu=True, pretrained_model=str(cp_model_path))
        else:
            raise RuntimeError("Detect environment name not for Cellpose. "
                                "Please follow the setup instructions provided at "
                                "'https://github.com/MouseLand/cellpose' "
                                "to create an environment.")
    
    """ Main Process """
    for img_path in img_paths:
        
        seg_dirname = f"{img_path.stem}.{seg_param_name}"
        seg_dir = img_path.parent.joinpath(f"{seg_desc}/{seg_dirname}")
        create_new_dir(seg_dir)
        
        # generate cell segmentation
        if seg_desc == "SLIC":
            seg1, seg2 = single_slic_labeling(seg_dir, img_path,
                                              n_segments, dark, merge,
                                              debug_mode)
        elif seg_desc == "Cellpose":
            seg1, seg2 = single_cellpose_prediction(seg_dir, img_path,
                                                    channels, cp_model, merge,
                                                    debug_mode)
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