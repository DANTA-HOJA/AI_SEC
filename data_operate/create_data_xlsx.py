import os
import sys
from pathlib import Path
import toml

abs_module_path = Path("./../modules/").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from logger import init_logger
from data.ProcessedDataInstance import ProcessedDataInstance

config_dir = Path( "./../Config/" ).resolve()

log = init_logger(r"Create Data Xlsx")

# -----------------------------------------------------------------------------------
# Load `(CreateXlsx)_data.toml`

with open(config_dir.joinpath("(CreateXlsx)_data.toml"), mode="r") as f_reader:
    config = toml.load(f_reader)
    
processed_inst_desc = config["data_processed"]["desc"]

# -----------------------------------------------------------------------------------
# Create `data.xlsx`

processed_data_instance = ProcessedDataInstance(config_dir, processed_inst_desc)
processed_data_instance.create_data_xlsx(log)