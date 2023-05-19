import os
import sys
from typing import List, Dict, Tuple
from colorama import Fore, Style
from pathlib import Path
import toml
from logging import Logger
from tqdm.auto import tqdm

import threading
from threading import Lock
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import traceback

sys.path.append("./../../modules/") # add path to scan customized module
from logger import init_logger
from gallery.utils import divide_in_grop
from gallery.cam.MtCamGalleryCreator import MtCamGalleryCreator


# -------------------------------------------------------------------------------------

log = init_logger(r"Make Cam Gallery")
config_path = Path(r"./../../Config/mk_cam_gallery.toml")
worker = 16
max_str_len_dict = {"fish_dsname": 0, "thread_name": 0}

# -------------------------------------------------------------------------------------

print(); log.info(f"Start {Fore.BLUE}`{os.path.basename(__file__)}`{Style.RESET_ALL}")
log.info(f"config_path: {Fore.GREEN}'{config_path.resolve()}'{Style.RESET_ALL}\n")

# -------------------------------------------------------------------------------------
""" get `fish_dsname_list` """
with open(config_path, mode="r") as f_reader:
    config = toml.load(f_reader)

load_dir_root = Path(config["model"]["history_root"])
model_name    = config["model"]["model_name"]
model_history = config["model"]["history"]

load_dir = load_dir_root.joinpath(model_name, model_history)
cam_result_root = load_dir.joinpath("cam_result")
cam_gallery_dir = load_dir.joinpath("!--- CAM Gallery")

assert not os.path.exists(cam_gallery_dir), f"dir: '{cam_gallery_dir}' already exists"

fish_dsname_list = [ str(path).split(os.sep)[-1] for path in list(cam_result_root.glob("*")) ] # will get dir_name likes `L_fish_111_A`
fish_dsname_list.sort()
# fish_dsname_list = fish_dsname_list[:16]

""" get `max_str_len` in `fish_dsname_list` """
for fish_dsname in fish_dsname_list:
    if len(fish_dsname) > max_str_len_dict["fish_dsname"]:
        max_str_len_dict["fish_dsname"] = len(fish_dsname)

""" divide `fish_dsname_list` for each worker """
fish_dsname_list_group = divide_in_grop(fish_dsname_list, worker)

# -------------------------------------------------------------------------------------

progressbars = [ tqdm(total=len(fish_dsname_list), desc="CAM Gallery ") for fish_dsname_list in fish_dsname_list_group ]

# -------------------------------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=worker)
lock = Lock()

def mt_cam_gallery(config_path:Path, fish_dsname_list:List[str], max_str_len_dict:int, 
                   lock:Lock, log:Logger, progressbar:tqdm):
    
    mt_cam_gallery_creator = MtCamGalleryCreator(config_path, max_str_len_dict, 
                                                   lock, log, progressbar)
    mt_cam_gallery_creator.gen_batch_cam_gallery(fish_dsname_list)


with executor:
    futures = [ executor.submit(mt_cam_gallery, 
                                config_path, fish_dsname_list, max_str_len_dict, 
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
