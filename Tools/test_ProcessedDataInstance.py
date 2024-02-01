import json
import os
import re
import sys
from pathlib import Path

from rich import print
from rich.panel import Panel
from rich.pretty import Pretty
from rich.traceback import install

abs_module_path = Path("./../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.config import get_coupled_config_name

install()
# -----------------------------------------------------------------------------/

# init `ProcessedDataInstance`
processed_data_instance = ProcessedDataInstance()
processed_data_instance.parse_config(get_coupled_config_name(__file__))

# reminder
temp_str = processed_data_instance.brightfield_processed_reminder
print(f"brightfield_processed_reminder: '{temp_str}'")
temp_str = processed_data_instance.palmskin_processed_reminder
print(f"palmskin_processed_reminder: '{temp_str}'\n")

# clustered_xlsx_files
temp_dict = processed_data_instance.clustered_files_dict
print(">>> Clustered Files [ {x} files ]".format(x=len(temp_dict)))
print(Panel(Pretty(temp_dict, expand_all=True)), "\n")

# alias_map
temp_dict = processed_data_instance.brightfield_processed_config
print(">>> Brightfield Analyze Config: ")
print(Panel(Pretty(temp_dict, expand_all=True)), "\n")
temp_dict = processed_data_instance.palmskin_processed_config
print(">>> Palmskin Preprocess Config: ")
print(Panel(Pretty(temp_dict, expand_all=True)), "\n")

# target results
rel_path, sorted_results_dict = processed_data_instance.get_sorted_results_dict("palmskin", "31_RGB_fusion.tif")
result_paths = list(sorted_results_dict.values())
print(">>> Relative Path to `dname_dir`: {} [ found {} items ]".format(rel_path, len(result_paths)))
print(Panel(Pretty(result_paths, expand_all=True)), "\n")