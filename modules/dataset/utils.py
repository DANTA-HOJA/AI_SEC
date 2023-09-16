import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

import cv2
from tomlkit.toml_document import TOMLDocument
# -----------------------------------------------------------------------------/


def gen_dataset_xlsx_name_dict(config:Union[dict, TOMLDocument]) -> Dict[str, str]:
    """ To generate dataset xlsx name corresponing to the parameters.
    
    config:
        cluster_desc (str):      e.g. '{SURF3C_KMeansLOG10_RND2022}_data.xlsx' ---> SURF3C \n
        crop_size (int):         e.g.                 512                      ---> CRPS512 \n
        shift_region (str):      e.g.                '1/4'                     ---> SF14 \n
        intensity (int):         e.g.                  20                      ---> INT20 \n
        drop_ratio (float):      e.g.                 0.3                      ---> DRP30 \n

    Raises:
        ValueError: If "Numerator of `shift_region` is not 1"

    Returns:
        str: e.g. `DS_SURF4C_CRPS512_SF14_INT20_DRP30`
    
    """
    cluster_desc: str = config["data_processed"]["cluster_desc"]
    crop_size: int = config["param"]["crop_size"]
    shift_region: str = config["param"]["shift_region"]
    intensity: int = config["param"]["intensity"]
    drop_ratio: float = config["param"]["drop_ratio"]
    
    """ Converting... `cluster_desc`  """
    cluster_desc_split = re.split("{|_|}", cluster_desc)
    match = re.search(r'\d+', cluster_desc_split[0]) # SURF3C
    if match:
        n_class = match.group()
    else:
        raise ValueError("Can't find any number in `clustered_xlsx_name`")
    # Name check
    sub_split = cluster_desc_split[0].split(n_class)
    if (sub_split[0] != "SURF") or (sub_split[1] != "C"):
        raise ValueError("First part of `cluster_desc` expect 'SURF[number]C', "
                         f"but got '{cluster_desc_split[0]}'. "
                         "Please check your config file.")
    
    """ Converting... `shift_region` """
    fraction = shift_region.split("/")
    assert (len(fraction) == 2) and (int(fraction[0]) == 1),  "Invalid format, expect '1/[denominator]'"
    fraction = f"{fraction[0]}{fraction[1]}"
    
    """ Converting... `drop_ratio` """
    ratio = int(drop_ratio*100)
    
    """ Create dict """
    temp_dict = {}
    temp_dict["prefix"]        = "DS"
    temp_dict["feature_class"] = cluster_desc_split[0]
    temp_dict["crop_size"]     = f"CRPS{crop_size}"
    temp_dict["shift_region"]  = f"SF{fraction}"
    temp_dict["intensity"]     = f"INT{intensity}"
    temp_dict["drop_ratio"]    = f"DRP{ratio}"
    
    return temp_dict
    # -------------------------------------------------------------------------/



def gen_crop_img(img:cv2.Mat, config:Union[dict, TOMLDocument]) -> List[cv2.Mat]:
    """ Generate the crop images using `crop_size` and `shift_region`

    Args:
        img (cv2.Mat): The source image to generate its crop images
    
    config:
        crop_size (int): Size/shape of cropped image
        shift_region (str): Offset distance between cropped images, \
                            e.g. if `shift_region` = '1/3', the overlap region of each cropped image is '2/3'

    Returns:
        List[cv2.Mat]
    """
    img_size = img.shape
    
    """ Get config variables """
    crop_size = config["param"]["crop_size"]
    shift_region = config["param"]["shift_region"]
    
    fraction = shift_region.split("/")
    assert (len(fraction) == 2) and (int(fraction[0]) == 1), ("Invalid format, `shift_region` expect '1/[denominator]', "
                                                              f"but got {shift_region}")
    DIV_PIECES = int(fraction[1]) # DIV_PIECES, abbr. of 'divide pieces'
    # print(DIV_PIECES, "\n")
    
    
    """ Calculate cropping step using `crop_size` """
    # Height
    quotient_h = int(img_size[0]/crop_size) # Height Quotient
    remainder_h = img_size[0]%crop_size # Height Remainder
    h_st_idx = remainder_h # 除不盡的部分從上方捨棄 ( image 上方為黑色 )
    h_crop_idxs = [(h_st_idx + int((crop_size/DIV_PIECES)*i)) 
                           for i in range(quotient_h*DIV_PIECES+1)]
    # print(f"h_crop_idxs = {h_crop_idxs}", f"len = {len(h_crop_idxs)}", "\n")
    
    # Width
    quotient_w = int(img_size[1]/crop_size) # Width Quotient
    remainder_w = img_size[1]%crop_size # Width Remainder
    w_st_idx = int(remainder_w/2) # 除不盡的部分兩側平分捨棄 ( image 左右都有黑色，平分 )
    w_crop_idxs = [(w_st_idx + int((crop_size/DIV_PIECES)*i))
                           for i in range(quotient_w*DIV_PIECES+1)]
    # print(f"w_crop_idxs = {w_crop_idxs}", f"len = {len(w_crop_idxs)}", "\n")
    
    
    """ Crop images """
    crop_img_list = []
    for i in range(len(h_crop_idxs)-DIV_PIECES):
        for j in range(len(w_crop_idxs)-DIV_PIECES):
            # print(h_crop_idxs[i], h_crop_idxs[i+DIV_PIECES], "\n", w_crop_idxs[j], w_crop_idxs[j+DIV_PIECES], "\n")
            crop_img_list.append(img[h_crop_idxs[i]:h_crop_idxs[i+DIV_PIECES], w_crop_idxs[j]:w_crop_idxs[j+DIV_PIECES], :])
    
    return crop_img_list
    # -------------------------------------------------------------------------/