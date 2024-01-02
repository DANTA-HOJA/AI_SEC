import os
from pathlib import Path
from typing import List, Union
import json

# import toml
# import tomlkit
# from tomlkit.toml_document import TOMLDocument

from .clioutput import CLIOutput

from ..assert_fn import *
from ..assert_fn import assert_run_under_repo_root, assert_only_1_config
# -----------------------------------------------------------------------------/


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
    # -------------------------------------------------------------------------/



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
    # -------------------------------------------------------------------------/



def get_repo_root(repo_name:str="ZebraFish_AP_POS", cli_out:CLIOutput=None) -> Path:
    """ Get repository root path on local machine

    Args:
        cli_out (CLIOutput, optional): a `CLIOutput` object. Defaults to None.

    Returns:
        Path: repository root
    """
    """ Analyze """
    path_split = os.path.abspath(".").split(os.sep)
    target_idx = get_target_str_idx_in_list(path_split, repo_name)
    assert_run_under_repo_root(target_idx)
    
    """ Generate path """
    repo_root = os.sep.join(path_split[:target_idx+1])
    
    """ CLI output """
    if cli_out: cli_out.write(f"Repository: '{repo_root}'")
    
    return Path(repo_root)
    # -------------------------------------------------------------------------/



def get_maxlength_in_dictkeys(eval_dict:dict) -> int:
    """ TODO
    """
    max_length = 0
    for key in eval_dict.keys():
        if len(key) > max_length:
            max_length = len(key)

    return max_length
    # -------------------------------------------------------------------------/



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
    # -------------------------------------------------------------------------/



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
    # -------------------------------------------------------------------------/



def formatter_padr0(obj) -> str:
    """ A string formatter, padding 0 at right ( depence on the maximum digit of len(obj) )
        - example: len(obj) = 100, return '03'
        - example: int(50), return '02' 
    """
    if hasattr(obj, "__len__"):
        # n_items: int = len(obj)
        # n_digit: str = str(n_items)
        # max_digit: int = len(n_digit)
        # formatter = f'0{max_digit}'
        return f"0{len(str(len(obj)))}"
    
    elif isinstance(obj, int):
        # assume `obj` = 50 (an integer)
        # str(50) = "50"
        # len("50") = 2
        # >>> 02
        return f"0{len(str(obj))}"
    
    else:
        raise NotImplementedError("Unrecognized type of given object")
    # -------------------------------------------------------------------------/



def exclude_tmp_paths(found_list:List[Path]):
    """ exclude paths if "tmp" or "temp" in path 
    """
    for _ in range(len(found_list)):
        path: Path = found_list.pop(0)
        path_split = str(path).split(os.sep)
        if ("tmp" not in path_split) and ("temp" not in path_split):
            found_list.append(path)
    
    return found_list
    # -------------------------------------------------------------------------/