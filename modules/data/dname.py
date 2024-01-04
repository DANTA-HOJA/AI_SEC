import filecmp
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

from ..shared.utils import get_target_str_idx_in_list
# -----------------------------------------------------------------------------/


def get_dname_sortinfo(string_with_fish_dname:Union[str, Path]) -> Tuple[int, str]:
    """ To extract `ID` and `Pos` from a provided string containing the (fish) dname, \
        where `ID` is a number and `Pos` is either 'A' or 'P'.
        
        dname example :
        - brightfield  : `20220610_CE001_palmskin_8dpf - Series001_fish_1_BF`
        - palmskin dname example : `20220610_CE001_palmskin_8dpf - Series001_fish_1_A_RGB`

    Args:
        string_with_fish_dname (Union[str, Path]): a string or a path including (fish) dname

    Raises:
        ValueError: Can't not match a proper directory name.
        TypeError: Input is not `str` or `Path` object.

    Returns:
        Tuple[int, str]: (ID, Position)
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
    # -------------------------------------------------------------------------/



def create_dict_by_id(path_list:List[Path]) -> Dict[int, Path]:
    """
    """
    return {get_dname_sortinfo(path)[0] : path for path in path_list}
    # -------------------------------------------------------------------------/



def merge_dict_by_id(auto_analysis_dict:Dict[int, Path], manual_analysis_dict:Dict[int, Path]):
    """
    """
    for key, value in manual_analysis_dict.items():
        auto_analysis_dict.pop(key, None)
        auto_analysis_dict[key] = manual_analysis_dict[key]
    return auto_analysis_dict
    # -------------------------------------------------------------------------/



def resave_result(original_path:Path, resave_dir:Path):
    """
    """
    if isinstance(original_path, Path): original_path:str = str(original_path)
    else: raise TypeError("'original_path' should be a 'Path' object, please using `from pathlib import Path`")
    
    if not isinstance(resave_dir, Path):
        raise TypeError("'resave_dir' should be a 'Path' object, please using `from pathlib import Path`")
    
    original_path_split = original_path.split(os.sep)
    
    if "_PalmSkin_preprocess" in original_path:
        target_idx = get_target_str_idx_in_list(original_path_split, "_PalmSkin_preprocess")
    elif "_BrightField_analyze" in original_path:
        target_idx = get_target_str_idx_in_list(original_path_split, "_BrightField_analyze")
    else:
        raise ValueError(f"Can't recognize the path: '{original_path}'")
    fish_dname = original_path_split[target_idx+1]
    
    file_ext = os.path.splitext(original_path)[-1]
    resave_path = resave_dir.joinpath(f"{fish_dname}{file_ext}")
    shutil.copy(original_path, resave_path)
    filecmp.cmp(original_path, resave_path)
    # -------------------------------------------------------------------------/