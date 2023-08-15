import os
import sys
import re
from pathlib import Path
from typing import List, Union
import json


__all__ = ["assert_is_pathobj", "assert_dir_exists", "assert_file_exists"]


def assert_only_1_config(found_list:List[Path], config_name:str):
    """ This assertion is for `load_config()` only

    Args:
        found_list (List[Path]): A result after running `Path.glob()`
        config_name (str): The name of config to find
    """
    assert len(found_list) > 0, f"Can't find any '{config_name}' ."
    assert len(found_list) == 1, f"Multiple config files, {json.dumps([str(path) for path in found_list], indent=2)}"


def assert_run_under_repo_root(target_idx:Union[int, None]):
    """ This assertion is for `get_repo_root()` only.

    Args:
        target_idx (_type_): A result after running `get_target_str_idx_in_list()`
    """
    assert target_idx is not None, ("Please switch your `WORKING_DIR` under this repository before run, "
                                    "and don't modify the name of this repository")


def assert_is_pathobj(path:Path):
    """ Check if `path` is a `Path` object 
    """
    assert isinstance(path, Path), ("The given path should be a `Path` object, "
                                    "please using `from pathlib import Path`")


def assert_dir_exists(dir:Path):
    """ 1. Check if `dir` is a `Path` object
        2. Check if `dir` exists
    """
    assert_is_pathobj(dir)
    assert dir.exists(), f"Can't find directory: '{dir.resolve()}'"


def assert_file_exists(file:Path):
    """ 1. Check if `file` is a `Path` object
        2. Check if `file` exists
    """
    assert_is_pathobj(file)
    assert file.exists(), f"Can't find file: '{file.resolve()}'"


def assert_lifname_split_in_4_part(name_split:List[str], lif_name:str):
    """
    """
    assert len(name_split) == 4, (f"file_name format error, current : '{lif_name}', "
                                  f"expect like : '20221125_AI005_palmskin_10dpf.lif'")


def assert_0_or_1_instance_root(found_list:List[Path], instance_desc:str):
    """ This assertion is for `get_instance_root()` only

    Args:
        found_list (List[Path]): A result after running `Path.glob()`
        instance_desc (str): The description of data instance to find
    """
    assert len(found_list) <= 1, (f"Found {len(found_list)} possible directories, "
                                  f"{json.dumps([str(path) for path in found_list], indent=2)} "
                                  f"{instance_desc} in `toml_config` is not unique")


def assert_0_or_1_processed_dir(found_list:List[Path], target_text:str):
    """ This assertion is for `get_processed_dir()` only

    Args:
        found_list (List[Path]): A result after running `Path.glob()`
    """
    assert len(found_list) <= 1, (f"{len(found_list)} directories are found, only one `{target_text}` is accepted. "
                                  f"Found: {json.dumps([str(path) for path in found_list], indent=2)}")