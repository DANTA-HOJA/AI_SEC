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
from modules.shared.utils import get_coupled_config_name

install()
# -----------------------------------------------------------------------------/

dataset_file = load_config(get_coupled_config_name(__file__))["path"]
print(f"\nDataset File: '{dataset_file}'\n")
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