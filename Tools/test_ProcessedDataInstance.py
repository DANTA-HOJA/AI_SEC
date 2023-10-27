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

from utils import get_tool_config

from modules.data.processeddatainstance import ProcessedDataInstance

install()
# -----------------------------------------------------------------------------/


# init `ProcessedDataInstance`
processed_data_instance = ProcessedDataInstance()
processed_data_instance.set_attrs(get_tool_config(__file__))

# reminder
temp_str = processed_data_instance.palmskin_processed_reminder
print(f"palmskin_processed_reminder: '{temp_str}'")
temp_str = processed_data_instance.brightfield_processed_reminder
print(f"brightfield_processed_reminder: '{temp_str}'\n")

# clustered_xlsx_files
temp_dict = processed_data_instance.clustered_xlsx_files_dict
print(">>> Clustered XLSX Files [ {x} files ]".format(x=len(temp_dict)))
print(Panel(Pretty(temp_dict, expand_all=True)), "\n")

# alias_map
temp_dict = processed_data_instance.brightfield_processed_alias_map
print(">>> Brightfield Alias Map [ {x} files ]".format(x=len(temp_dict)))
print(Panel(Pretty(temp_dict, expand_all=True)), "\n")
temp_dict = processed_data_instance.palmskin_processed_alias_map
print(">>> Palmskin Alias Map [ {x} files ]".format(x=len(temp_dict)))
print(Panel(Pretty(temp_dict, expand_all=True)), "\n")

# target results
rel_path, result_paths = processed_data_instance._get_sorted_results("palmskin", "RGB_fusion")
print(">>> Relative Path to `dname_dir`: {} [ found {} items ]".format(rel_path, len(result_paths)))
print(Panel(Pretty(result_paths, expand_all=True)), "\n")