import sys
from pathlib import Path

abs_module_path = Path("./../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.shared.utils import get_repo_root
from modules.data.processeddatainstance import ProcessedDataInstance

""" Detect Repository """
repo_root = get_repo_root(display_on_CLI=True)


processed_data_instance = ProcessedDataInstance()
processed_data_instance.load_config("0.5.create_data_xlsx.toml")

""" Main Process """
processed_data_instance.create_data_xlsx()