import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from collections import Counter
import json

import pandas as pd

abs_module_path = Path("./../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from utils import get_debug_config
from modules.data.dataset import dsname
from modules.shared.config import load_config
# -----------------------------------------------------------------------------/


dataset_xlsx_path = load_config(get_debug_config(__file__))["path"]
print(f"\nDataset XLSX Path: '{dataset_xlsx_path}'\n")
dataset_xlsx_df: pd.DataFrame = pd.read_excel(dataset_xlsx_path, engine='openpyxl')

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
print(f">>> Full Discard Test Fish [ {len(test_fish_dict)} fish ]: \n"
      f"{json.dumps(list(test_fish_dict.keys()), indent=2)}\n")