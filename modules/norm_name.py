import os
import re
from typing import Tuple
import filecmp



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



def resave_result(original_path:str, output_dir:str, result:str):
    if "MetaImage" in original_path: fish_name = original_path.split(os.sep)[-3]
    else: fish_name = original_path.split(os.sep)[-2]
    file_ext = result.split(".")[-1]
    out_path = os.path.join(output_dir, f"{fish_name}.{file_ext}")
    os.system(f"copy \"{original_path}\" \"{out_path}\" ")
    filecmp.cmp(original_path, out_path)