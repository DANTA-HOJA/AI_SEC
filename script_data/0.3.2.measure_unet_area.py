import sys
from pathlib import Path

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.lif.brightfieldunetareameter import BrightfieldUNetAreaMeter
from modules.shared.utils import get_repo_root
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

brightfield_unet_area_meter = BrightfieldUNetAreaMeter()
brightfield_unet_area_meter.run("0.3.1.analyze_brightfield.toml")