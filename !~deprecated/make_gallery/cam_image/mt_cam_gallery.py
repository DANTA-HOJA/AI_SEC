import os
import sys
import traceback
from typing import List, Dict, Tuple
from colorama import Fore, Style
from pathlib import Path
from logging import Logger
from tqdm.auto import tqdm
import toml
import matplotlib

from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

rel_module_path = "./../../modules/"
sys.path.append( str(Path(rel_module_path).resolve()) ) # add path to scan customized module

from logger import init_logger
from gallery.utils import divide_in_grop
from gallery.cam.MtCamGalleryCreator import MtCamGalleryCreator

config_dir = Path( "./../../Config/" ).resolve()

matplotlib.use('agg')
log = init_logger(r"Make Cam Gallery")

# -------------------------------------------------------------------------------------
# Load `db_path_plan.toml`

with open(config_dir.joinpath("db_path_plan.toml"), mode="r") as f_reader:
    dbpp_config = toml.load(f_reader)

db_root = Path(dbpp_config["root"])

# -------------------------------------------------------------------------------------
config_name = "(MakeGallery)_cam.toml"

config_path = config_dir.joinpath(config_name)
with open(config_path, mode="r") as f_reader:
    config = toml.load(f_reader)

model_history = config["model_prediction"]["history"]
worker = config["multiprocessing"]["worker"]

# -----------------------------------------------------------------------------------
# Generate `path_vars`

load_dir_root = db_root.joinpath(dbpp_config["model_prediction"])
load_dir = load_dir_root.joinpath(model_history)
cam_result_root = load_dir.joinpath("cam_result")
cam_gallery_dir = load_dir.joinpath("!--- CAM Gallery")

assert cam_result_root.exists(), f"Can't find 'cam_result' directory: '{cam_result_root}'"
assert not cam_gallery_dir.exists(), f"Directory '!--- CAM Gallery' already exists: '{cam_gallery_dir}'"

# -------------------------------------------------------------------------------------
# Multi-Thread instance

executor = ThreadPoolExecutor(max_workers=worker)
lock = Lock()

def mt_cam_gallery(config_dir:Path, config_name:str, fish_dsname_list:List[str], max_str_len_dict:int, 
                   lock:Lock, log:Logger, progressbar:tqdm):
    
    mt_cam_gallery_creator = MtCamGalleryCreator(config_dir, config_name, max_str_len_dict, 
                                                 lock, log, progressbar)
    mt_cam_gallery_creator.gen_batch_cam_gallery(fish_dsname_list)

# -------------------------------------------------------------------------------------
""" get `fish_dsname_list` """

fish_dsname_list = [ str(path).split(os.sep)[-1] for path in list(cam_result_root.glob("*")) ] # will get dir_name likes `L_fish_111_A`
fish_dsname_list.sort()
# fish_dsname_list = fish_dsname_list[:16]
max_str_len_dict = {"fish_dsname": 0, "thread_name": 0}

""" get `max_str_len` in `fish_dsname_list` """
for fish_dsname in fish_dsname_list:
    if len(fish_dsname) > max_str_len_dict["fish_dsname"]:
        max_str_len_dict["fish_dsname"] = len(fish_dsname)

""" divide `fish_dsname_list` for each worker """
fish_dsname_list_group = divide_in_grop(fish_dsname_list, worker)

# -------------------------------------------------------------------------------------

print(); log.info(f"Run {Fore.BLUE}`{os.path.basename(__file__)}`{Style.RESET_ALL}")
log.info(f"config_path: {Fore.GREEN}'{config_path.resolve()}'{Style.RESET_ALL}\n")

progressbars = [ tqdm(total=len(fish_dsname_list), desc="CAM Gallery ") for fish_dsname_list in fish_dsname_list_group ]

with executor:
    futures = [ executor.submit(mt_cam_gallery, 
                                config_dir, config_name, fish_dsname_list, max_str_len_dict, 
                                lock, log, progressbars[i]) \
                for i, fish_dsname_list in enumerate(fish_dsname_list_group) ]
    
    try:
        for future in concurrent.futures.as_completed(futures):
            future.result()
        
    except Exception:
        for progressbar in progressbars: progressbar.close()
        print(); print(f"{traceback.format_exc()}") # 輸出異常訊息
    
    else:
        for progressbar in progressbars: progressbar.close()
        print(); log.info(f"Done {Fore.BLUE}`{os.path.basename(__file__)}`{Style.RESET_ALL}\n")
