import os
import sys
import re
from pathlib import Path
from typing import List


def assert_only_1_config(found_list:List[Path], config_name:str):
    assert len(found_list) > 0, f"Can't find any '{config_name}' ."
    assert len(found_list) == 1, f"Multiple config files, {found_list}"


def assert_run_under_repo_root(target_idx):
    assert target_idx is not None, ("Please switch your shell under this repository before run, "
                                    "and don't modify the name of this repository")


def assert_is_pathobj(dir:Path):
    assert isinstance(dir, Path), ("The given path should be a `Path` object, "
                                   "please using `from pathlib import Path`")