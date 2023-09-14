import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

import cv2
from tomlkit.toml_document import TOMLDocument
# -----------------------------------------------------------------------------/


def gen_dataset_xlsx_name(config:Union[dict, TOMLDocument], dict_format:bool=False) -> Union[dict, str]:
    """ To generate dataset xlsx name corresponing to the parameters.
    
    config:
        cluster_desc (str):           e.g. "{SURF3C_KMeansLOG10_RND2022}_data.xlsx" ---> SURF3C \n
        crop_size (int):                e.g.                 512                      ---> CRPS512 \n
        shift_region (str):             e.g.                '1/4'                     ---> SF14 \n
        intensity (int):                e.g.                  20                      ---> INT20 \n
        drop_ratio (float):             e.g.                 0.3                      ---> DRP30 \n

    Raises:
        ValueError: If "Numerator of `shift_region` is not 1"

    Returns:
        str: e.g. `DS_SURF4C_CRPS512_SF14_INT20_DRP30`
    
    """
    cluster_desc = config["data_processed"]["cluster_desc"]
    crop_size = config["param"]["crop_size"]
    shift_region = config["param"]["shift_region"]
    intensity = config["param"]["intensity"]
    drop_ratio = config["param"]["drop_ratio"]
    
    """ Converting... `cluster_desc`  """
    cluster_desc_split = re.split("{|_|}", cluster_desc)
    match = re.search(r'\d+', cluster_desc_split[0]) # SURF3C
    if match:
        n_class = match.group()
    else:
        raise ValueError("Can't find any number in `clustered_xlsx_name`")
    # name check
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
    
    """ create dict """
    temp_dict = {}
    temp_dict["prefix"]        = "DS"
    temp_dict["feature_class"] = cluster_desc_split[0]
    temp_dict["crop_size"]     = f"CRPS{crop_size}"
    temp_dict["shift_region"]  = f"SF{fraction}"
    temp_dict["intensity"]     = f"INT{intensity}"
    temp_dict["drop_ratio"]    = f"DRP{ratio}"
    
    if dict_format == True: return temp_dict
    else: return "_".join(list(temp_dict.values()))
    # -------------------------------------------------------------------------/