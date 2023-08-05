import os
import sys
import re
from pathlib import Path
from typing import List


__all__ = ["assert_is_pathobj", "assert_dir_exists"]


def assert_only_1_config(found_list:List[Path], config_name:str):
    """ This assertion is for `load_config()` only

    Args:
        found_list (List[Path]): A result after running `Path.glob()`
        config_name (str): The name of config to find
    """
    assert len(found_list) > 0, f"Can't find any '{config_name}' ."
    assert len(found_list) == 1, f"Multiple config files, {found_list}"


def assert_run_under_repo_root(target_idx):
    """ This assertion is for `get_repo_root()` only.

    Args:
        target_idx (_type_): A result after running `get_target_str_idx_in_list()`
    """
    assert target_idx is not None, ("Please switch your shell under this repository before run, "
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