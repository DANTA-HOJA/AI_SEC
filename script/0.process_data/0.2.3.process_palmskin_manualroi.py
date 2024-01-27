import sys
from pathlib import Path

abs_module_path = Path("./../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.data.lif.palmskinmanualroicreator import PalmskinManualROICreator
from modules.shared.utils import get_repo_root
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

palmskin_manualroi_creator = PalmskinManualROICreator()
palmskin_manualroi_creator.run("0.2.3.process_palmskin_manualroi.toml")