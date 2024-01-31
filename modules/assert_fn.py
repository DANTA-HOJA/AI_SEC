import os
import sys
import re
from pathlib import Path
from typing import List, Union
import json


__all__ = ["assert_is_pathobj", "assert_dir_exists", "assert_file_exists", 
           "assert_dir_not_exists", "assert_file_not_exists"]
# -----------------------------------------------------------------------------/


def assert_only_1_config(found_list:List[Path], config_name:str):
    """ This assertion is for `load_config()` only

    Args:
        found_list (List[Path]): A result after running `Path.glob()`
        config_name (str): The name of config to find
    """
    assert len(found_list) > 0, f"Can't find any '{config_name}' ."
    assert len(found_list) == 1, (f"Multiple config files, "
                                 f"{json.dumps([str(path) for path in found_list], indent=2)}")
    # -------------------------------------------------------------------------/



def assert_run_under_repo_root(target_idx:Union[int, None]):
    """ This assertion is for `get_repo_root()` only.

    Args:
        target_idx (_type_): A result after running `get_target_str_idx_in_list()`
    """
    assert target_idx is not None, ("Please switch your `WORKING_DIR` under this repository before run, "
                                    "and don't modify the name of this repository")
    # -------------------------------------------------------------------------/



def assert_is_pathobj(path:Path):
    """ Check `path` is a `Path` object 
    """
    if not isinstance(path, Path):
        raise TypeError("The given path should be a `Path` object, "
                        "please using `from pathlib import Path`")
    # -------------------------------------------------------------------------/



def assert_dir_exists(dir:Path):
    """ 1. Check `dir` is a `Path` object
        2. Check `dir` exists
    """
    assert_is_pathobj(dir)
    if not dir.exists():
        raise FileNotFoundError(f"Can't find directory: '{dir.resolve()}'")
    # -------------------------------------------------------------------------/



def assert_file_exists(file:Path):
    """ 1. Check `file` is a `Path` object
        2. Check `file` exists
    """
    assert_is_pathobj(file)
    if not file.exists():
        raise FileNotFoundError(f"Can't find file: '{file.resolve()}'")
    # -------------------------------------------------------------------------/



def assert_dir_not_exists(dir:Path):
    """ 1. Check `dir` is a `Path` object
        2. Check `dir` not exists
    """
    assert_is_pathobj(dir)
    if dir.exists():
        raise FileExistsError(f"Directory already exists: '{dir.resolve()}'")
    # -------------------------------------------------------------------------/



def assert_file_not_exists(file:Path):
    """ 1. Check `file` is a `Path` object
        2. Check `file` not exists
    """
    assert_is_pathobj(file)
    if file.exists():
        raise FileNotFoundError(f"File already exists: '{file.resolve()}'")
    # -------------------------------------------------------------------------/



def assert_lifname_split_in_4_part(name_split:List[str], lif_name:str):
    """
    """
    assert len(name_split) == 4, (f"File name format error, current : '{lif_name}', "
                                  f"expect like : '20221125_AI005_palmskin_10dpf.lif'")
    # -------------------------------------------------------------------------/



def assert_0_or_1_instance_root(found_list:List[Path], instance_desc:str):
    """ This assertion is for `get_instance_root()` only

    Args:
        found_list (List[Path]): A result after running `Path.glob()`
        instance_desc (str): The description of data instance to find
    """
    assert len(found_list) <= 1, (f"Found {len(found_list)} possible directories, "
                                  f"`{instance_desc}` in config is not unique. "
                                  f"Directories: {json.dumps([str(path) for path in found_list], indent=2)}")
    # -------------------------------------------------------------------------/



def assert_0_or_1_processed_dir(found_list:List[Path], target_text:str):
    """ This assertion is for `get_processed_dir()` only

    Args:
        found_list (List[Path]): A result after running `Path.glob()`
    """
    assert len(found_list) <= 1, (f"Found {len(found_list)} possible directories, "
                                  f"only one `{target_text}` is accepted. "
                                  f"Directories: {json.dumps([str(path) for path in found_list], indent=2)}")
    # -------------------------------------------------------------------------/



def assert_0_or_1_recollect_dir(found_list:List[Path], target_text:str):
    """ This assertion is for `get_recollect_dir()` only

    Args:
        found_list (List[Path]): A result after running `Path.glob()`
    """
    assert len(found_list) <= 1, (f"Found {len(found_list)} possible directories, "
                                  f"only one `{target_text}` is accepted. "
                                  f"Directories: {json.dumps([str(path) for path in found_list], indent=2)}") 
    # -------------------------------------------------------------------------/



def assert_0_or_1_history_dir(found_list:List[Path], time_stamp:str, state:str=None):
    """ This assertion is for `BaseImageTester._set_history_dir()` only
    """
    if state is None: temp_str = f"`{time_stamp}`"
    else: temp_str = f"`{time_stamp}` and `{state}`"
    
    assert len(found_list) <= 1, (f"Found {len(found_list)} possible directories, "
                                  f"'dirname' combine with {temp_str} in config is not unique. "
                                  f"Directories: {json.dumps([str(path) for path in found_list], indent=2)}")
    # -------------------------------------------------------------------------/



def assert_file_ext(file_name:str, target_ext: str):
    """

    Args:
        file_name (str): a file name with extension
        target_ext (str): the file extension to check, example: ".jpg"
    """    
    file_ext = os.path.splitext(Path(file_name).parts[-1])[1]
    
    assert file_ext == target_ext, (f"'{file_name}' is not a '{target_ext}' file")
    # -------------------------------------------------------------------------/