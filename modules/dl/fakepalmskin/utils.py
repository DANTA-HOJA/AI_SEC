import colorsys
import random

import numpy as np
from matplotlib import colors as mcolors
from skimage.color import rgb2lab
from skimage.segmentation import mark_boundaries
# -----------------------------------------------------------------------------/


def gen_singlecolor_palmskin(seg:np.ndarray,
                             cytosol_color: tuple[float, float, float]=(0.5, 0.5, 0.5),
                             border_color: tuple[float, float, float]=(1.0, 1.0, 1.0),
                             bg_color: tuple[float, float, float]=None,
                             ) -> np.ndarray:
    """
    """
    fake_palmskin = np.full((*seg.shape, 3), cytosol_color, dtype=np.float64)
    
    if bg_color is not None:
        fake_palmskin[seg == 0] = bg_color
    
    fake_palmskin = mark_boundaries(fake_palmskin, seg, color=border_color)

    return fake_palmskin
    # -------------------------------------------------------------------------/


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