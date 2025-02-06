import sys
from pathlib import Path

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.dl.tester.imagetester import VitB16AOnlyImageTester
from modules.shared.utils import get_repo_root
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

vit_b_16_aonly_image_tester = VitB16AOnlyImageTester()
vit_b_16_aonly_image_tester.run("3.test_by_image.toml")