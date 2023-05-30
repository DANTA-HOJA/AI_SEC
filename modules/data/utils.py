import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union



def get_fish_ID_pos(string_with_fish_Dname:Union[str, Path]) -> Tuple[int, str]:
    """
    string_with_fish_Dname:
    
        - bf:  .../`20220610_CE001_palmskin_8dpf - Series001_fish_1_BF`/[file_with_extension]
        - rgb: .../`20220610_CE001_palmskin_8dpf - Series001_fish_1_A_RGB`/[file_with_extension]
    
    """
    
    if isinstance(string_with_fish_Dname, Path):
        string_with_fish_Dname = str(string_with_fish_Dname)
        if "MetaImage" in string_with_fish_Dname: fish_name = string_with_fish_Dname.split(os.sep)[-3]
        elif "reCollection" in string_with_fish_Dname: fish_name = string_with_fish_Dname.split(os.sep)[-1]
        else: fish_name = string_with_fish_Dname.split(os.sep)[-2]
    elif isinstance(string_with_fish_Dname, str): 
        fish_name = string_with_fish_Dname
    else: 
        raise TypeError("unrecognized type of `string_with_fish_Dname`. Only `pathlib.Path` or `str` are accepted.")
    
    fish_name_split = re.split(" |_|-", fish_name)
    if "BF" in fish_name: assert len(fish_name_split) == 10, f"len(fish_name_list) = '{len(fish_name_split)}', expect '10' "
    if "RGB" in fish_name: assert len(fish_name_split) == 11, f"len(fish_name_list) = '{len(fish_name_split)}', expect '11' "
    return int(fish_name_split[8]), fish_name_split[9]



def create_dict_by_fishID(path_list:List[Path]) -> Dict[int, Path]:
    return {get_fish_ID_pos(path)[0] : path for path in path_list}



def merge_BF_analysis(auto_analysis_dict:Dict[int, Path], manual_analysis_dict:Dict[int, Path]):
    for key, value in manual_analysis_dict.items():
        auto_analysis_dict.pop(key, None)
        auto_analysis_dict[key] = manual_analysis_dict[key]
    return auto_analysis_dict