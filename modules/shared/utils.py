import os
from pathlib import Path
from typing import List, Union
from logging import Logger
from tqdm.auto import tqdm
import json

import toml
import tomlkit
from tomlkit.toml_document import TOMLDocument

from .clioutput import CLIOutput

from ..assert_fn import *
from ..assert_fn import assert_run_under_repo_root, assert_only_1_config



def create_new_dir(dir:Union[str, Path], msg_end:str="\n", cli_out:CLIOutput=None) -> None:
    """ If `dir` is not exist then create it.

    Args:
        dir (Union[str, Path]): a path
        msg_end (str, optional): control the end of message shows on CLI. Defaults to [NewLine].
        cli_out (CLIOutput, optional): a `CLIOutput` object. Defaults to None.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        """ CLI output """
        if cli_out: cli_out.write(f"Directory: '{dir}' is created!{msg_end}")



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



def get_repo_root(cli_out:CLIOutput=None) -> Path:
    """ Get repository root path on local machine

    Args:
        cli_out (CLIOutput, optional): a `CLIOutput` object. Defaults to None.

    Returns:
        Path: repository root
    """
    """ Analyze """
    path_split = os.path.abspath(".").split(os.sep)
    target_idx = get_target_str_idx_in_list(path_split, "ZebraFish_AP_POS")
    assert_run_under_repo_root(target_idx)
    
    """ Generate path """
    repo_root = os.sep.join(path_split[:target_idx+1])
    
    """ CLI output """
    if cli_out: cli_out.write(f"Repository: '{repo_root}'")
    
    return Path(repo_root)



def load_config(config_file:Union[str, Path], reserve_comment:bool=False,
                cli_out:CLIOutput=None) -> Union[dict, TOMLDocument]:
    """ Scan and load the specific config under repo root

    Args:
        config_file (str): full file name, like `abc.toml`
        reserve_comment (bool, optional): Defaults to False.
        cli_out (CLIOutput, optional): a `CLIOutput` object. Defaults to None.

    Returns:
        Union[dict, TOMLDocument]: a toml config
    """
    if reserve_comment:
        load_fn = tomlkit.load
    else:
        load_fn = toml.load
    
    path = None
    if isinstance(config_file, Path):
        path = config_file
    elif isinstance(config_file, str):
        repo_root = get_repo_root()
        found_list = list(repo_root.glob(f"**/{config_file}"))
        assert_only_1_config(found_list, config_file)
        path = found_list[0]
    else:
        raise NotImplementedError("Argument `config_file` should be `str` or `Path` object.")
    
    """ CLI output """
    if cli_out: cli_out.write(f"Config Path: '{path}'")
    
    with open(path, mode="r") as f_reader:
        config = load_fn(f_reader)
    
    return config



def get_maxlength_in_dictkeys(eval_dict:dict) -> int:
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
            maxlen = get_maxlength_in_dictkeys(obj)
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