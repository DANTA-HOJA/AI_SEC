from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ..shared.config import dump_config, load_config
# -----------------------------------------------------------------------------/


def count_area(seg:np.ndarray) -> tuple[int, str]:
    """ unit: pixel
    """
    foreground = seg > 0
    return int(np.sum(foreground)), "area"
    # -------------------------------------------------------------------------/


def count_element(seg:np.ndarray, key:str) -> tuple[int, str]:
    """
    """
    return len(np.unique(seg[seg > 0])), f"{key}_count"
    # -------------------------------------------------------------------------/


def count_average_size(analysis_dict:dict[str, Any], key:str) -> tuple[float, str]:
    """ unit: pixel
    """
    area = analysis_dict["area"]
    element_cnt = analysis_dict[f"{key}_count"]
    
    if element_cnt == 0:
        average_size = 0
    else:
        average_size = float(round(area/element_cnt, 5))
    
    return  average_size, f"{key}_avg_size"
    # -------------------------------------------------------------------------/


def get_max_patch_size(seg:np.ndarray) -> tuple[int, str]:
    """ (deprecated) unit: pixel
    
        Note: this function is replaced by `get_patch_sizes()`
    """
    max_size = 0
    
    labels = np.unique(seg)
    for label in labels:
        if label != 0:
            bw = (seg == label)
            size = np.sum(bw)
            if size > max_size:
                max_size = size
    
    return int(max_size), "max_patch_size"
    # -------------------------------------------------------------------------/


def get_patch_sizes(seg:np.ndarray) -> tuple[list[int], str]:
    """ unit: pixel
    """
    tmp = seg.flatten().tolist()
    tmp = Counter(tmp)
    
    try: # remove background
        tmp.pop(0)
    except KeyError:
        print("Warning: background label missing")
    
    tmp = dict(sorted(tmp.items(), reverse=True, key=lambda x: x[1]))
    
    return list(tmp.values()), f"patch_sizes"
    # -------------------------------------------------------------------------/


def update_seg_analysis_dict(analysis_dict:dict[str, Any], value:Any, key:str) -> dict[str, Any]:
    """
    """
    analysis_dict = deepcopy(analysis_dict)
    analysis_dict[key] = value
    
    return  analysis_dict
    # -------------------------------------------------------------------------/


def update_ana_toml_file(toml_path:Path, analysis_dict:dict):
    """
    """
    if toml_path.exists():
        save_dict = load_config(toml_path)
        for k, v in analysis_dict.items():
            save_dict[k] = v
    else:
        save_dict = analysis_dict
    
    dump_config(toml_path, save_dict)
    # -------------------------------------------------------------------------/