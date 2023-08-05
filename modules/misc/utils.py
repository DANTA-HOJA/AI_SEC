import os
from pathlib import Path
from typing import List, Union
from logging import Logger
from tqdm.auto import tqdm
import json

import toml
import tomlkit
from tomlkit.toml_document import TOMLDocument

from ..assert_fn import *



def decide_cli_output(logger:Logger=None):
    """ decide to use `print` or `logger.info` as CLI output
    """
    if logger:
        cli_out = logger.info
    else:
        cli_out = print
    
    return cli_out



def create_new_dir(dir:Union[str, Path], end:str="\n", 
                   display_in_CLI:bool=True, logger:Logger=None) -> None:
    """if `path` is not exist then create it.

    Args:
        path (Union[str, Path]): a path
        end (str, optional): control the end of string show on CLI. Defaults to "\n".
        display_in_CLI (bool, optional): whether to print on CLI. Defaults to True.
        use_tqdm (bool, optional): if the script using `tqdm` turn this on. Defaults to False.
    """
    cli_out = decide_cli_output(logger)
    
    if not os.path.exists(dir):
        os.makedirs(dir)
        if display_in_CLI:
            cli_out(f"Directory: '{dir}' is created!{end}")



def get_target_str_idx_in_list(source_list:List[str], target_str:str) -> Union[int, None]:
    """ TODO
    """
    target_idx = None
    
    for i, text in enumerate(source_list):
        if target_str in text:
            if target_idx is None:
                target_idx = i
            else:
                raise ValueError(f"Too many '{target_str}' in list")
    
    return target_idx



def get_repo_root(logger:Logger=None) -> Path:
    """ TODO
    """
    cli_out = decide_cli_output(logger)
    
    path_split = os.path.abspath(".").split(os.sep)
    target_idx = get_target_str_idx_in_list(path_split, "ZebraFish_AP_POS")
    assert_run_under_repo_root(target_idx)
    
    """ generate path """
    repo_root = Path(os.sep.join(path_split[:target_idx+1]))
    cli_out(f"Repository: '{repo_root}'")
    
    return repo_root



def load_config(config_name:str, reserve_comment:bool=False, logger:Logger=None) -> Union[dict, TOMLDocument]:
    """ TODO
    """
    cli_out = decide_cli_output(logger)
    
    if reserve_comment:
        load_fn = tomlkit.load
    else:
        load_fn = toml.load    
    
    repo_root = get_repo_root()
    found_list = list(repo_root.glob(f"**/{config_name}"))
    assert_only_1_config(found_list, config_name)
    cli_out(f"Config Path: '{found_list[0]}'")
    
    with open(found_list[0], mode="r") as f_reader:
        config = load_fn(f_reader)
        
    return config



def get_maxlength_of_dictkeys(eval_dict:dict) -> int:
    """ TODO
    """
    max_length = 0
    for key in eval_dict.keys():
        if len(key) > max_length:
            max_length = len(key)

    return max_length



def get_attr_formatstring(object:object, attr:str) -> Union[str, TypeError]:
    """get attribute in formatted string, format = "self.{attr} : {value}"

    Args:
        object (object): an object
        attr (str): attribute in the object

    Returns:
        str: formatted string
    """
    obj = getattr(object, attr)
    
    if not callable(obj):
        if isinstance(obj, dict):
            """ An align dict for pretty CLI output """
            align_dict = {}
            maxlen = get_maxlength_of_dictkeys(obj)
            for key, value in obj.items():
                align_dict[f"{key:^{maxlen+2}}"] = value
            """ json format """
            formatted = json.dumps(align_dict, indent=4)
            return f"self.{attr} : {formatted}"
        else:
            return f"self.{attr} : {obj}"
    else:
        return TypeError("Attribute is a function (callable).")



def print_sorted_attrs(object:object, filter_magic_fn:bool=True):
    """ show object's attributes
    """
    print("="*80)
    
    attributes = dir(object)
    
    if filter_magic_fn:
        attributes = [attr for attr in attributes if not attr.startswith('__')]

    for attr in attributes:
        string = get_attr_formatstring(object, attr)
        if string is not TypeError:
            print(string)
    
    print("="*80)