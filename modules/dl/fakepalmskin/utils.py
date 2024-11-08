import colorsys
import random

import numpy as np
from matplotlib import colors as mcolors
from skimage.color import rgb2lab
from skimage.segmentation import mark_boundaries
# -----------------------------------------------------------------------------/


def gen_singlecolor_palmskin(seg: np.ndarray,
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


def gen_unique_random_color_pool(n_labels: list,
                                 existing_color_pool: dict=None,
                                 lab_lthres: float=None) -> dict:
    """Range of `lab_lthres`: [0.0, 100.0]

    Args:
        n_labels (list): How many colors should generate
        existing_color_pool (dict, optional): If given, will extend the given color pool. Defaults to None.
        hsv_vthreshold (int, optional): Only collect the colors over this threshold. Defaults to None.

    Returns:
        dict: {Hex rgb: float rgb}
    """
    if lab_lthres is not None:
        if not isinstance(lab_lthres, float):
            raise ValueError(f"`lab_lthres` is not `float`, got {lab_lthres}")
        if not ((lab_lthres >= 0.0) and ((lab_lthres <= 100.0))):
            raise ValueError(f"Range of `lab_lthres`: [0.0, 100.0], got {lab_lthres}")
    
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
        
        if lab_lthres is None:
            color_pool[f"{hex_rgb}"] = float_rgb # save color
        else:
            lab_color = rgb2lab(float_rgb) # range of L: [0.0, 100.0]
            if lab_color[0] > lab_lthres:
                color_pool[f"{hex_rgb}"] = float_rgb # save color
    
    color_pool.pop("#000000") # remove `black` (background)

    return color_pool
    # -------------------------------------------------------------------------/