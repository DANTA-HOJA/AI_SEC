import sys
from pathlib import Path

from rich.traceback import install

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.dl.tester.fishtester import VitB16AonlyNoCropFishTester
from modules.shared.utils import get_repo_root

install()
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

vit_b_16_aonly_nocrop_fish_tester = VitB16AonlyNoCropFishTester()
vit_b_16_aonly_nocrop_fish_tester.run("4.test_by_fish.toml")