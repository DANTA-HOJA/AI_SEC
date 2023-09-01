import sys
from pathlib import Path

abs_module_path = Path("./../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.shared.utils import get_repo_root, load_config
from modules.data.processeddatainstance import ProcessedDataInstance

""" Detect Repository """
repo_root = get_repo_root(display_on_CLI=True)

""" Get variable """
config = load_config("0.4.collect_results.toml")
image_type = config["collection"]["image_type"]
result_alias = config["collection"]["result_alias"]
log_mode = config["collection"]["log_mode"]


processed_data_instance = ProcessedDataInstance()
processed_data_instance.load_config("0.4.collect_results.toml")
processed_data_instance.collect_results(image_type, result_alias, log_mode)