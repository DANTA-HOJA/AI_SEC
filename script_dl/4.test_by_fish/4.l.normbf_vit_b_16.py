import sys
from pathlib import Path

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.dl.tester.fishtester import VitB16NormBFFishTester
from modules.shared.config import get_batch_config, get_batch_config_arg
from modules.shared.utils import exclude_tmp_paths, get_repo_root
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

vit_b_16_normbf_fish_tester = VitB16NormBFFishTester()
args = get_batch_config_arg()

if args.batch_mode == True:
    
    config_paths = sorted(exclude_tmp_paths(get_batch_config(__file__)))
    for config_path in config_paths:
        vit_b_16_normbf_fish_tester.run(config_path)

else: vit_b_16_normbf_fish_tester.run("4.test_by_fish.toml")