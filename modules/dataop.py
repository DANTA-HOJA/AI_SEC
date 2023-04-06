import os
import re
from typing import Tuple, Dict



def get_fish_ID_pos(path:str) -> Tuple[int, str]:
    # bf:  .../20220610_CE001_palmskin_8dpf - Series001_fish_1_BF/[file_with_extension]
    # rgb: .../20220610_CE001_palmskin_8dpf - Series001_fish_1_A_RGB/[file_with_extension]
    if "MetaImage" in path: fish_name = path.split(os.sep)[-3]
    elif "reCollection" in path: fish_name = path.split(os.sep)[-1]
    else: fish_name = path.split(os.sep)[-2]
    fish_name_list = re.split(" |_|-", fish_name)
    if "BF" in fish_name: assert len(fish_name_list) == 10, f"len(fish_name_list) = '{len(fish_name_list)}', expect '10' "
    if "RGB" in fish_name: assert len(fish_name_list) == 11, f"len(fish_name_list) = '{len(fish_name_list)}', expect '11' "
    return int(fish_name_list[8]), fish_name_list[9]



def create_dict_by_fishID(path_list:list) -> Dict[int, str]:
    return {get_fish_ID_pos(path)[0] : path for path in path_list}



def merge_BF_analysis(auto_analysis_dict:Dict[int, str], manual_analysis_dict:Dict[int, str]):
    for key, value in manual_analysis_dict.items():
        auto_analysis_dict.pop(key, None)
        auto_analysis_dict[key] = manual_analysis_dict[key]
    return auto_analysis_dict