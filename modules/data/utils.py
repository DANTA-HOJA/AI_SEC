import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

abs_module_path = str(Path("./../modules/").resolve())
if abs_module_path not in sys.path: sys.path.append(abs_module_path) # add path to scan customized module

from utils import get_target_str_idx_in_list


def get_fish_id_pos(string_with_fish_dname:Union[str, Path]) -> Tuple[int, str]:
    """
    string_with_fish_dname:
    
        - bf:  .../`20220610_CE001_palmskin_8dpf - Series001_fish_1_BF`/[file_with_extension]
        - rgb: .../`20220610_CE001_palmskin_8dpf - Series001_fish_1_A_RGB`/[file_with_extension]
    
    """
    if isinstance(string_with_fish_dname, Path):
        
        string_with_fish_dname = str(string_with_fish_dname)
        string_with_fish_dname_split = string_with_fish_dname.split(os.sep)
        
        if "_reCollection" in string_with_fish_dname:
            target_idx = get_target_str_idx_in_list(string_with_fish_dname_split, "_reCollection")
            fish_name = string_with_fish_dname_split[target_idx+2]
        else:
            if "_PalmSkin_preprocess" in string_with_fish_dname:
                target_idx = get_target_str_idx_in_list(string_with_fish_dname_split, "_PalmSkin_preprocess")
            elif "_BrightField_analyze" in string_with_fish_dname:
                target_idx = get_target_str_idx_in_list(string_with_fish_dname_split, "_BrightField_analyze")
            else:
                raise ValueError(f"Can't recognize the path: '{string_with_fish_dname}'")
            fish_name = string_with_fish_dname_split[target_idx+1]
        
    elif isinstance(string_with_fish_dname, str):
        fish_name = string_with_fish_dname
    else:
        raise TypeError("unrecognized type of `string_with_fish_dname`. Only `pathlib.Path` or `str` are accepted.")
    
    fish_name = fish_name.split(".")[0]
    fish_name_split = re.split(" |_|-", fish_name)
    if "BF" in fish_name: assert len(fish_name_split) == 10, f"len(fish_name_list) = '{len(fish_name_split)}', expect '10' "
    if "RGB" in fish_name: assert len(fish_name_split) == 11, f"len(fish_name_list) = '{len(fish_name_split)}', expect '11' "
    
    return int(fish_name_split[8]), fish_name_split[9]



def create_dict_by_fishID(path_list:List[Path]) -> Dict[int, Path]:
    return {get_fish_id_pos(path)[0] : path for path in path_list}



def merge_BF_analysis(auto_analysis_dict:Dict[int, Path], manual_analysis_dict:Dict[int, Path]):
    for key, value in manual_analysis_dict.items():
        auto_analysis_dict.pop(key, None)
        auto_analysis_dict[key] = manual_analysis_dict[key]
    return auto_analysis_dict