import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
from logging import Logger
import json

from ..shared.utils import decide_cli_output, get_attr_formatstring
from ..assert_fn import *


class LIFNameChecker():
    
    def __init__(self) -> None:
        """ A class for checking the image name in LIF file
        """
        self._name_split:list = []
        self.format_and_type:str = ""
        self._failed_cnt:int = 0
        self.check_dict:dict = {}
    
    

    def _check_len_of_name_split(self, target_len:int) -> None:
        """ check length of `self._name_split`
        """
        dict_key = "len(image_name_split)"
        length = len(self._name_split)
        
        if length != target_len:
            self._failed_cnt = -1 # CRITICAL_ERROR
            self.check_dict["failed message"] = f"CRITICAL_ERROR --> len(image_name_split) = {length}, expect {target_len} " # WARNING:
        else: 
            self.check_dict[dict_key] = "PASS"



    def _check_Series_format(self, check_pos:int) -> None:
        """ spelling of 'Series'
        """
        dict_key = "spelling of 'Series'"
        temp_list = re.split("[0-9]", self._name_split[check_pos])
        
        if temp_list[0] != "Series":
            self._failed_cnt += 1
            self.check_dict[dict_key] = f"FAILED --> '{temp_list[0]}', misspelling of 'Series' " # WARNING:
            self.check_dict["Series[num]"] = "NOT CHECK"
            return None
        else:
            self.check_dict[dict_key] = "PASS"
        
        """ check postfix 'Series[num]'
        """
        dict_key = "Series[num]"
        temp_list = self._name_split[check_pos].split("Series")
        
        if len(temp_list[1]) != 3:
            self._failed_cnt += 1
            self.check_dict[dict_key] = f"FAILED --> detect {len(temp_list[1])} digits, expect 3 digits " # WARNING:
        else:
            self.check_dict[dict_key] = "PASS"



    def _check_fish_spelling(self, check_pos:int) -> None:
        """ spelling of 'fish'
        """
        dict_key = "spelling of 'fish'"
        if self._name_split[check_pos] != "fish":
            self._failed_cnt += 1
            self.check_dict[dict_key] = f"FAILED --> '{self._name_split[check_pos]}', misspelling of 'fish' " # WARNING:
        else: 
            self.check_dict[dict_key] = "PASS"



    def _check_fishid_format(self, check_pos:int) -> None:
        """ check fish id is a number
        """
        dict_key = "fish_[ID]"
        try:
            int(self._name_split[check_pos])
            self.check_dict[dict_key] = "PASS"
        except Exception as e:
            self._failed_cnt += 1
            self.check_dict[dict_key] = f"FAILED --> {e} " # WARNING:



    def _check_palmskin_spelling(self, check_pos:int) -> None:
        """ spelling of 'palmskin'
        """
        dict_key = "spelling of 'palmskin'"
        if self._name_split[check_pos] != "palmskin": 
            self._failed_cnt += 1
            self.check_dict[dict_key] = f"FAILED --> '{self._name_split[check_pos]}', misspelling of 'palmskin' " # WARNING:
        else: 
            self.check_dict[dict_key] = "PASS"



    def _check_dpf_format(self, check_pos:int) -> None:
        """ spelling of 'dpf'
        """
        dict_key = "spelling of 'dpf'"
        temp_list = re.split("[0-9]", self._name_split[check_pos])
        
        if temp_list[-1] != "dpf":
            self._failed_cnt += 1
            self.check_dict[dict_key] = f"FAILED --> '{temp_list[-1]}', misspelling of 'dpf' " # WARNING:
            self.check_dict["[num]_dpf"] = "NOT CHECK"
            return None
        else:
            self.check_dict[dict_key] = "PASS"
        
        """ check postfix '[num]_dpf'
        """
        dict_key = "[num]_dpf"
        temp_list = self._name_split[check_pos].split("dpf")
        try: 
            int(temp_list[0])
            self.check_dict[dict_key] = "PASS"
        except Exception as e: 
            self._failed_cnt += 1
            self.check_dict[dict_key] = f"FAILED --> {e} " # WARNING:



    def _check_A_P_format(self, check_pos:int) -> None:
        """ check fish position label is 'A' or 'P'
        """
        dict_key = "A or P"
        if self._name_split[check_pos] != "A" and self._name_split[check_pos] != "P": 
            self._failed_cnt += 1
            self.check_dict[dict_key] = f"FAILED --> '{self._name_split[check_pos]}', expect 'A' or 'P' " # WARNING:
        else: 
            self.check_dict[dict_key] = "PASS"



    def _check_RGB_spelling(self, check_pos:int) -> None:
        """ spelling of 'RGB'
        """
        dict_key = "spelling of 'RGB'"
        if self._name_split[check_pos] != "RGB":
            self._failed_cnt += 1
            self.check_dict[dict_key] = f"FAILED --> '{self._name_split[check_pos]}', misspelling of 'RGB' " # WARNING:
        else: 
            self.check_dict[dict_key] = "PASS"



    def _check_BF_spelling(self, check_pos:int) -> None:
        """ spelling of 'BF'
        """ 
        dict_key = "spelling of 'BF'"
        if self._name_split[check_pos] != "BF": 
            self._failed_cnt += 1
            self.check_dict[dict_key] = f"FAILED --> '{self._name_split[check_pos]}', misspelling of 'BF' " # WARNING:
        else: 
            self.check_dict[dict_key] = "PASS"
    
    
    
    def _calcu_failed_cnt(self) -> None:
        """ calculate `self._failed_cnt` and write message to `self.check_dict
        """
        failed_key_list = [key for key, value in self.check_dict.items() if value.split(" --> ")[0] == "FAILED"]
        dict_key = "failed message"
        self.check_dict[dict_key] = (f"ERROR --> image_name, At least {self._failed_cnt} failed: "
                                     f"{', '.join(failed_key_list)}")
    
    
    
    def _reset_attrs(self):
        """ reset all attribute
        """
        self._name_split = []
        self.format_and_type = ""
        self._failed_cnt = 0
        self.check_dict = {}
    
    
    
    def check_image_name(self, image_name:str, format_and_type:str) -> None:
        """
        Args:
            image_name (str):
            - name example:
            
                - `20220617_CE002_palmskin_8dpf - Series001_fish_11_palmskin_8dpf` (old_bf)
                - `20221127_AI005_palmskin_12dpf - Series005_fish_207_BF` (new_bf)
                - `20220610_CE001_palmskin_8dpf - Series001_fish_1_palmskin_8dpf_A` (old_rgb)
                - `20221125_AI005_palmskin_10dpf - Series001_fish_165_A_RGB` (new_rgb)
            format_and_type (str): `old_rgb`, `new_rgb`, `old_bf`, `new_bf`

        Returns:
            _type_: _description_
        """
        self._reset_attrs()
        self.format_and_type = format_and_type
        self._name_split = re.split(" |_|-", image_name)
        
        if self.format_and_type == "old_rgb":   self._check_len_of_name_split(6)
        elif self.format_and_type == "new_rgb": self._check_len_of_name_split(5)
        elif self.format_and_type == "old_bf":  self._check_len_of_name_split(5)
        elif self.format_and_type == "new_bf":  self._check_len_of_name_split(4)
        
        if self._failed_cnt == -1: 
            return ValueError

        self._check_Series_format(0)
        self._check_fish_spelling(1)
        self._check_fishid_format(2)
        
        if self.format_and_type == "old_rgb":
            self._check_palmskin_spelling(3)
            self._check_dpf_format(4)
            self._check_A_P_format(5)
        
        if self.format_and_type == "new_rgb":
            self._check_A_P_format(3)
            self._check_RGB_spelling(4)
        
        if self.format_and_type == "old_bf":
            self._check_palmskin_spelling(3)
            self._check_dpf_format(4)
            
        if self.format_and_type == "new_bf":
            self._check_BF_spelling(3)

        if self._failed_cnt > 0 :
            self._calcu_failed_cnt()
    
    
    
    def __repr__(self):
        """
        """
        output = f"{'='*80}\n" # CLI divider 1
        
        for attr in self.__dict__:
            string = get_attr_formatstring(self, attr)
            if string is not TypeError:
                output += f"{string}\n"
        
        output += f"{'='*80}\n" # CLI divider 2
        
        return output



def scan_lifs_under_dir(dir:Path, batches:list, logger:Logger=None) -> List[str]:
    """ Scan Leica LIF file
    """
    assert_dir_exists(dir)
    cli_out = decide_cli_output(logger)
    lif_path_list = []
    
    if batches:
        for batch in batches:
            found_list = list(dir.joinpath(batch).glob("**/*.lif"))
            lif_path_list.extend(found_list)
    else:
        lif_path_list = list(dir.glob("**/*.lif"))
    
    lif_path_list = [str(lif_path) for lif_path in lif_path_list]
    lif_path_list.sort(key=lambda x: x.split(os.sep)[-1])
    
    """ CLI output """
    formatted = json.dumps(lif_path_list, indent=4)
    cli_out(f'lif_path_list {type(lif_path_list)}: {formatted}')
    cli_out(f"[ found {len(lif_path_list)} lif files ]")
    
    return lif_path_list