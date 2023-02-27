import re
from typing import *
import json


def check_len_of_name_list(name_list:List[str], target_len:int, failed_cnt:int, check_dict:Dict[str, str]) -> Tuple[int, Dict[str, str]]:
    dict_key = "len(image_name_list)"
    
    if len(name_list) != target_len:
        failed_cnt += 1
        check_dict[f"{dict_key:^24}"] = f"FAILED --> len(image_name_list) = '{len(name_list)}', expect '{target_len}' " # WARNING:
    else: 
        check_dict[f"{dict_key:^24}"] = "PASS"
    
    return failed_cnt, check_dict



def check_Series_format(name_list:List[str], check_pos:int, failed_cnt:int, check_dict:Dict[str, str]) -> Tuple[int, Dict[str, str]]:
    
    # spelling of 'Series'
    dict_key = "spelling of 'Series'"
    temp_list = re.split("[0-9]", name_list[check_pos])
    if temp_list[0] != "Series":
        failed_cnt += 1
        check_dict[f"{dict_key:^24}"] = f"FAILED --> '{temp_list[0]}', misspelling of 'Series' " # WARNING:
        check_dict[f"{'Series[num]':^24}"] = "NOT CHECK"
        return failed_cnt, check_dict
    else:
        check_dict[f"{dict_key:^24}"] = "PASS"
    
    # Series[num]
    dict_key = "Series[num]"
    try: 
        temp_list = name_list[check_pos].split("Series")
        num = int(temp_list[1])
        assert len(temp_list[1]) == 3, f"only {len(temp_list[1])} digits, expect 3 digits "
        check_dict[f"{dict_key:^24}"] = "PASS"
    except Exception as e: 
        failed_cnt += 1
        check_dict[f"{dict_key:^24}"] = f"FAILED --> {e} " # WARNING:
    
    return failed_cnt, check_dict



def check_fish_format(name_list:List[str], check_pos:int, failed_cnt:int, check_dict:Dict[str, str]) -> Tuple[int, Dict[str, str]]:

    dict_key = "spelling of 'fish'"
    if name_list[check_pos] != "fish": 
        failed_cnt += 1
        check_dict[f"{dict_key:^24}"] = f"FAILED --> '{name_list[check_pos]}', misspelling of 'fish' " # WARNING:
    else: 
        check_dict[f"{dict_key:^24}"] = "PASS"
        
    return failed_cnt, check_dict



def check_fishID_format(name_list:List[str], check_pos:int, failed_cnt:int, check_dict:Dict[str, str]) -> Tuple[int, Dict[str, str]]:
    
    dict_key = "fish_[ID]"
    try:
        int(name_list[check_pos])
        check_dict[f"{dict_key:^24}"] = "PASS"
    except Exception as e:
        failed_cnt += 1
        check_dict[f"{dict_key:^24}"] = f"FAILED --> {e} " # WARNING:
    
    return failed_cnt, check_dict



def check_palmskin_format(name_list:List[str], check_pos:int, failed_cnt:int, check_dict:Dict[str, str]) -> Tuple[int, Dict[str, str]]:
    
    dict_key = "spelling of 'palmskin'"
    if name_list[check_pos] != "palmskin": 
        failed_cnt += 1
        check_dict[f"{dict_key:^24}"] = f"FAILED --> '{name_list[check_pos]}', misspelling of 'palmskin' " # WARNING:
    else: 
        check_dict[f"{dict_key:^24}"] = "PASS"
    
    return failed_cnt, check_dict



def check_dpf_format(name_list:List[str], check_pos:int, failed_cnt:int, check_dict:Dict[str, str]) -> Tuple[int, Dict[str, str]]:

    # spelling of 'dpf'
    dict_key = "spelling of 'dpf'"
    temp_list = re.split("[0-9]", name_list[check_pos])
    if temp_list[-1] != "dpf":
        failed_cnt += 1
        check_dict[f"{dict_key:^24}"] = f"FAILED --> '{temp_list[-1]}', misspelling of 'dpf' " # WARNING:
        check_dict[f"{'[num]_dpf':^24}"] = "NOT CHECK"
        return failed_cnt, check_dict
    else:
        check_dict[f"{dict_key:^24}"] = "PASS"
    
    # [num]_dpf
    dict_key = "[num]_dpf"
    try: 
        temp_list = name_list[check_pos].split("dpf")
        num = int(temp_list[0])
        check_dict[f"{dict_key:^24}"] = "PASS"
    except Exception as e: 
        failed_cnt += 1
        check_dict[f"{dict_key:^24}"] = f"FAILED --> {e} " # WARNING:
    
    return failed_cnt, check_dict



def check_A_P_format(name_list:List[str], check_pos:int, failed_cnt:int, check_dict:Dict[str, str]) -> Tuple[int, Dict[str, str]]:
    
    dict_key = "A or P"
    if name_list[check_pos] != "A" and name_list[check_pos] != "P": 
        failed_cnt += 1
        check_dict[f"{dict_key:^24}"] = f"FAILED --> '{name_list[check_pos]}', expect 'A' or 'P' " # WARNING:
    else: 
        check_dict[f"{dict_key:^24}"] = "PASS"

    return failed_cnt, check_dict



def check_RGB_format(name_list:List[str], check_pos:int, failed_cnt:int, check_dict:Dict[str, str]) -> Tuple[int, Dict[str, str]]:
    
    dict_key = "spelling of 'RGB'"
    if name_list[check_pos] != "RGB": 
        failed_cnt += 1
        check_dict[f"{dict_key:^24}"] = f"FAILED --> '{name_list[check_pos]}', misspelling of 'RGB' " # WARNING:
    else: 
        check_dict[f"{dict_key:^24}"] = "PASS"

    return failed_cnt, check_dict



def check_rgbimage_name(image_name:str, format_and_type:str) -> Tuple[int, Dict[str, str]]:

        # rgb image name example:
        #
        # 20220610_CE001_palmskin_8dpf - Series001_fish_1_palmskin_8dpf_A (old format)
        # 20221125_AI005_palmskin_10dpf - Series001_fish_165_A_RGB (new format)
        
        split_list = re.split(" |_|-", str(image_name))
        # print(split_list)
        check_dict = {}
        check_dict[f"{'format_and_type':^24}"] = f"{format_and_type}"
        failed_cnt = 0
        
        if format_and_type == "old_rgb": failed_cnt, check_dict = check_len_of_name_list(split_list, 6, failed_cnt, check_dict)
        if format_and_type == "new_rgb": failed_cnt, check_dict = check_len_of_name_list(split_list, 5, failed_cnt, check_dict)
        if failed_cnt == 1: return failed_cnt, check_dict

        
        failed_cnt, check_dict = check_Series_format(split_list, 0, failed_cnt, check_dict)
        failed_cnt, check_dict = check_fish_format(split_list, 1, failed_cnt, check_dict)
        failed_cnt, check_dict = check_fishID_format(split_list, 2, failed_cnt, check_dict)
        
        if format_and_type == "old_rgb":
            failed_cnt, check_dict = check_palmskin_format(split_list, 3, failed_cnt, check_dict)
            failed_cnt, check_dict = check_dpf_format(split_list, 4, failed_cnt, check_dict)
            failed_cnt, check_dict = check_A_P_format(split_list, 5, failed_cnt, check_dict)
        
        if format_and_type == "new_rgb": 
            failed_cnt, check_dict = check_A_P_format(split_list, 3, failed_cnt, check_dict)
            failed_cnt, check_dict = check_RGB_format(split_list, 4, failed_cnt, check_dict)
        
        # print(failed_cnt)
        # print(json.dumps(check_dict, indent=2))

        return failed_cnt, check_dict