import os
import sys
from typing import List, Dict, Tuple
from colorama import Fore, Style
from pathlib import Path
import toml

from tqdm.auto import tqdm
from multiprocessing import Pool, Manager
import traceback

sys.path.append("./../../modules/") # add path to scan customized module
from logger import init_logger
from gallery.utils import divide_in_grop
from gallery.cam.MpCamGalleryCreator import MpCamGalleryCreator

# -------------------------------------------------------------------------------------

def mp_cam_gallery(config_path:Path, fish_dsname_list:List[str],
                   max_str_len_queues, lock,
                   init_sig_queue, pbar_sig_queue,
                   error_sig_queue):
    
    mp_cam_gallery_creator = None
    
    try:
        mp_cam_gallery_creator = MpCamGalleryCreator(config_path, max_str_len_queues, lock,
                                                     init_sig_queue, pbar_sig_queue)
        mp_cam_gallery_creator.gen_batch_cam_gallery(fish_dsname_list)
        
    except Exception:
        
        if mp_cam_gallery_creator: error_sig_queue.put(mp_cam_gallery_creator.process_name)
        if mp_cam_gallery_creator: error_sig_queue.put(mp_cam_gallery_creator.process_id)
        with lock: error_sig_queue.put(traceback.format_exc())

# -------------------------------------------------------------------------------------

if __name__ == "__main__":

    log = init_logger(r"Make Cam Gallery")
    config_path = Path(r"./../../Config/mk_cam_gallery.toml")
    max_processes = 16
    max_str_len_dict = {"fish_dsname": 0, "process_name": 0}

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
    fish_dsname_list_group = divide_in_grop(fish_dsname_list, max_processes)

    # -------------------------------------------------------------------------------------

    progressbars = [ tqdm(total=len(fish_dsname_list), desc="CAM Gallery ") for fish_dsname_list in fish_dsname_list_group ]

    # -------------------------------------------------------------------------------------
    pool = Pool(processes=max_processes)
    manager = Manager()
    lock = manager.Lock()
    max_str_len_queues = manager.Queue(); max_str_len_queues.put(max_str_len_dict)
    init_sig_queues    = [ manager.Queue() for _ in fish_dsname_list_group ]
    pbar_sig_queues    = [ manager.Queue() for _ in fish_dsname_list_group ]
    error_sig_queues   = [ manager.Queue() for _ in fish_dsname_list_group ]


    for i, fish_dsname_list in enumerate(fish_dsname_list_group):
        kwds = {
            "config_path": config_path, 
            "fish_dsname_list": fish_dsname_list, 
            "max_str_len_queues": max_str_len_queues,
            "lock": lock, 
            "init_sig_queue": init_sig_queues[i],
            "pbar_sig_queue": pbar_sig_queues[i],
            "error_sig_queue": error_sig_queues[i],
        }
        pool.apply_async(mp_cam_gallery, kwds=kwds)

    pool.close() # 不再接受新任務


    try:
        error_process_idx = 0
        complete_cnt = 0
        while complete_cnt < max_processes:
            for i in range(max_processes):
                
                if not init_sig_queues[i].empty():
                    with lock:
                        log.info(init_sig_queues[i].get())
                        log.info(init_sig_queues[i].get())
                
                if not pbar_sig_queues[i].empty():
                    with lock:
                        pbar_sig = pbar_sig_queues[i].get()
                        if isinstance(pbar_sig, str): progressbars[i].desc = pbar_sig
                        if isinstance(pbar_sig, int): progressbars[i].update(pbar_sig)
                        progressbars[i].refresh()
                        if (progressbars[i].n / progressbars[i].total) == 1.0: 
                            complete_cnt += 1
                    
                if not error_sig_queues[i].empty():
                    error_process_idx = i
                    raise RuntimeError
    
    
    except (Exception, KeyboardInterrupt) as e:
        for progressbar in progressbars: progressbar.close()
        print()
        if isinstance(e, KeyboardInterrupt): print("KeyboardInterrupt\n")
        if isinstance(e, Exception):
            while not error_sig_queues[error_process_idx].empty():
                msg = str(error_sig_queues[error_process_idx].get())
                print(f"\n{msg}") if "Traceback" in msg else log.warning(msg)
    
    else:
        for progressbar in progressbars: progressbar.close()
        pool.join() # 等待所有 sub-process 完成
        print(); log.info(f"Done {Fore.BLUE}`{os.path.basename(__file__)}`{Style.RESET_ALL}\n")