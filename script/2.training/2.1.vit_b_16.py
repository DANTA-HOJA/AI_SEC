import sys
from pathlib import Path

abs_module_path = Path("./../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.dl.trainer.vitb16trainer import VitB16Trainer
from modules.shared.config import get_batch_config, get_batch_config_arg
from modules.shared.utils import get_repo_root

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")


vit_b_16_trainer = VitB16Trainer()
args = get_batch_config_arg()

if args.batch_mode == True:
    
    config_paths = sorted(get_batch_config(__file__))
    for config_path in config_paths:
        vit_b_16_trainer.run(config_path)

else: vit_b_16_trainer.run()