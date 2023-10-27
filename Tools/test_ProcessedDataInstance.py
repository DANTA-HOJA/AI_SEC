import os
import sys
import re
from pathlib import Path
import json

abs_module_path = Path("./../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from utils import get_tool_config
from modules.data.processeddatainstance import ProcessedDataInstance
# -----------------------------------------------------------------------------/


# init `ProcessedDataInstance`
processed_data_instance = ProcessedDataInstance()
processed_data_instance.set_attrs(get_tool_config(__file__))

# reminder
print(f"palmskin_processed_reminder: '{processed_data_instance.palmskin_processed_reminder}'")
print(f"brightfield_processed_reminder: '{processed_data_instance.brightfield_processed_reminder}'\n")

# clustered_xlsx_files
temp_dict = {key: str(value) for key, value in processed_data_instance.clustered_xlsx_files_dict.items()}
print(f">>> Clustered XLSX Files: \n{json.dumps(temp_dict, indent=2)}\n")

# alias_map
alias_map = processed_data_instance.brightfield_processed_alias_map
print(f">>> Brightfield Alias Map [ {len(alias_map)} files ]: \n{json.dumps(alias_map, indent=2)}\n")
alias_map = processed_data_instance.palmskin_processed_alias_map
print(f">>> Palmskin Alias Map [ {len(alias_map)} files ]: \n{json.dumps(alias_map, indent=2)}\n")

# target results
rel_path, result_paths = processed_data_instance._get_sorted_results("palmskin", "RGB_fusion")
result_paths = [str(result_path) for result_path in result_paths]
print(f">>> Relative Path to `dname_dir`: {rel_path} [ found {len(result_paths)} items ] \n{json.dumps(result_paths, indent=2)}\n")