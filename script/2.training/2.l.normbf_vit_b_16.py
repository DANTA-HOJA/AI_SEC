import sys
from pathlib import Path

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.dl.trainer import VitB16NormBFTrainer
from modules.shared.config import get_batch_config, get_batch_config_arg
from modules.shared.utils import exclude_tmp_paths, get_repo_root

import matplotlib; matplotlib.use("agg")
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

vit_b_16_normbf_trainer = VitB16NormBFTrainer()
args = get_batch_config_arg()

if args.batch_mode == True:
    
    config_paths = sorted(exclude_tmp_paths(get_batch_config(__file__)))
    for config_path in config_paths:
        vit_b_16_normbf_trainer.run(config_path)

else: vit_b_16_normbf_trainer.run("2.training.toml")