import sys
from pathlib import Path

abs_module_path = Path("./../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.data.lif.batchlifnamechecker import BatchLIFNameChecker
from modules.shared.utils import get_repo_root

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")


batch_lif_name_checker = BatchLIFNameChecker()
batch_lif_name_checker.run("0.1.check_lif_name.toml")