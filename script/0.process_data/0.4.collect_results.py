import os
import sys
from pathlib import Path
import toml

abs_module_path = Path("./../modules/").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from data.ProcessedDataInstance import ProcessedDataInstance

config_dir = Path( "./../Config/" ).resolve()

# -----------------------------------------------------------------------------------
config_name = "(CollectResults)_data.toml"

with open(config_dir.joinpath(config_name), mode="r") as f_reader:
    config = toml.load(f_reader)

processed_inst_desc = config["data_processed"]["desc"]

processed_name = config["collection"]["processed_name"]
result_alias = config["collection"]["result_alias"]
log_mode = config["collection"]["log_mode"]

# -----------------------------------------------------------------------------------
# Create collection

processed_data_instance = ProcessedDataInstance(config_dir, processed_inst_desc)
processed_data_instance.collect_results(processed_name, result_alias, log_mode)