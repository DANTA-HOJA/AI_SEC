import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rich import print
from rich.panel import Panel
from rich.pretty import Pretty
from rich.traceback import install

abs_module_path = Path("./../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.data.dataset import dsname
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator

install()
# -----------------------------------------------------------------------------/

# load config
config = load_config("dataset.toml")
print(Pretty(config, expand_all=True))

dataset_seed_dir: str = config["dataset"]["seed_dir"]
dataset_data: str = config["dataset"]["data"]
dataset_palmskin_result: str = config["dataset"]["palmskin_result"]
dataset_base_size: str = config["dataset"]["base_size"]
dataset_classif_strategy: str = config["dataset"]["classif_strategy"]
dataset_file_name: str = config["dataset"]["file_name"]

# read dataset file (.csv)
path_navigator = PathNavigator()
dataset_cropped = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")

dataset_file = dataset_cropped.joinpath(dataset_seed_dir,
                                        dataset_data,
                                        dataset_palmskin_result,
                                        dataset_base_size,
                                        dataset_classif_strategy,
                                        dataset_file_name)

dataset_xlsx_df: pd.DataFrame = pd.read_csv(dataset_file, encoding='utf_8_sig')

# all test fish
test_df = dataset_xlsx_df[(dataset_xlsx_df["dataset"] == "test")]
test_fish = sorted(Counter(test_df["parent (dsname)"]).keys(), 
                   key=dsname.get_dsname_sortinfo)
test_fish_dict = {fish: fish for fish in test_fish}

# preserve test fish
preserve_test_df = test_df[(test_df["state"] == "preserve")]
preserve_test_fish = sorted(Counter(preserve_test_df["parent (dsname)"]).keys(),
                            key=dsname.get_dsname_sortinfo)
preserve_test_fish_dict = {fish: fish for fish in preserve_test_fish}

# find discard test fish
for fish in preserve_test_fish_dict.keys():
    test_fish_dict.pop(fish)

# CLI out
print(">>> Full Discard Test Fish [ {x} files ]".format(x=len(test_fish_dict)))
print(Panel(Pretty(list(test_fish_dict.keys()), expand_all=True)), "\n")