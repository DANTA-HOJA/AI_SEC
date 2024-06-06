import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from slic_labeling import run_single_slic_analysis

from modules.data.dataset.dsname import get_dsname_sortinfo
from modules.shared.clioutput import CLIOutput
from modules.shared.config import (dump_config, get_coupled_config_name,
                                   load_config)
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, get_repo_root

install()
# -----------------------------------------------------------------------------/


def count_area(seg:np.ndarray) -> tuple[int, str]:
    """ unit: pixel
    """
    foreground = seg > 0
    return int(np.sum(foreground)), "area"
    # -------------------------------------------------------------------------/


def count_element(seg:np.ndarray, key:str) -> tuple[int, str]:
    """
    """
    return len(np.unique(seg[seg > 0])), f"{key}_count"
    # -------------------------------------------------------------------------/


def count_average_size(analysis_dict:dict[str, Any], key:str) -> tuple[float, str]:
    """ unit: pixel
    """
    analysis_dict = deepcopy(analysis_dict)
    area = analysis_dict["area"]
    element_cnt = analysis_dict[f"{key}_count"]
    
    if element_cnt == 0:
        average_size = 0
    else:
        average_size = float(round(area/element_cnt, 5))
    
    return  average_size, f"{key}_avg_size"
    # -------------------------------------------------------------------------/


def get_max_patch_size(seg:np.ndarray) -> tuple[int, str]:
    """ (deprecated) unit: pixel
    
        Note: this function is replaced by `get_patch_size()`
    """
    max_size = 0
    
    labels = np.unique(seg)
    for label in labels:
        if label != 0:
            bw = (seg == label)
            size = np.sum(bw)
            if size > max_size:
                max_size = size
    
    return int(max_size), "max_patch_size"
    # -------------------------------------------------------------------------/


def get_patch_sizes(seg:np.ndarray) -> tuple[list[int], str]:
    """ unit: pixel
    """
    tmp = seg.flatten().tolist()
    tmp = Counter(tmp)
    
    try: # remove background
        tmp.pop(0)
    except KeyError:
        print("Warning: background label missing")
    
    tmp = dict(sorted(tmp.items(), reverse=True, key=lambda x: x[1]))
    
    return list(tmp.values()), f"patch_sizes"
    # -------------------------------------------------------------------------/


def update_slic_analysis_dict(analysis_dict:dict[str, Any], value:Any, key:str) -> dict[str, Any]:
    """
    """
    analysis_dict = deepcopy(analysis_dict)
    analysis_dict[key] = value
    
    return  analysis_dict
    # -------------------------------------------------------------------------/


def update_toml_file(toml_path:Path, analysis_dict:dict):
    """
    """
    if toml_path.exists():
        save_dict = load_config(toml_path)
        for k, v in analysis_dict.items():
            save_dict[k] = v
    else:
        save_dict = analysis_dict
    
    dump_config(toml_path, save_dict)
    # -------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")

    """ Init components """
    path_navigator = PathNavigator()
    cli_out = CLIOutput()
    cli_out.divide()

    # load config
    # `dark` and `merge` are two parameters as color space distance, determined by experiences
    config = load_config("get_cell_feature.toml")
    # [dataset]
    dataset_seed_dir: str = config["dataset"]["seed_dir"]
    dataset_data: str = config["dataset"]["data"]
    dataset_palmskin_result: str = config["dataset"]["palmskin_result"]
    dataset_base_size: str = config["dataset"]["base_size"]
    # [SLIC]
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    merge: int       = config["SLIC"]["merge"]
    debug_mode: bool = config["SLIC"]["debug_mode"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()

    """ Colloct image file names """
    dataset_cropped: Path = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
    src_root = dataset_cropped.joinpath(dataset_seed_dir,
                                        dataset_data,
                                        dataset_palmskin_result,
                                        dataset_base_size)
    paths = sorted(src_root.glob("*/*/*.tiff"), key=get_dsname_sortinfo)
    print(f"Total files: {len(paths)}")

    """ Apply SLIC on each image """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(paths))
        
        for path in paths:
            
            result_name = path.stem
            dname_dir = path.parents[0]
            slic_dir = dname_dir.joinpath(f"SLIC/{result_name}_{{dark_{dark}}}")
            create_new_dir(slic_dir)
            
            print(f"[ {dname_dir.parts[-1]} : '{slic_dir}' ]")
            dump_config(slic_dir.joinpath(f"{{copy}}_{Path(__file__).stem}.toml"), config)
            
            # SLIC
            cell_seg, patch_seg = run_single_slic_analysis(slic_dir, path,
                                                            n_segments, dark, merge,
                                                            debug_mode)
            
            # update
            analysis_dict = {}
            analysis_dict = update_slic_analysis_dict(analysis_dict, *count_area(cell_seg))
            analysis_dict = update_slic_analysis_dict(analysis_dict, *count_element(cell_seg, "cell"))
            analysis_dict = update_slic_analysis_dict(analysis_dict, *count_element(patch_seg, "patch"))
            analysis_dict = update_slic_analysis_dict(analysis_dict, *count_average_size(analysis_dict, "cell"))
            analysis_dict = update_slic_analysis_dict(analysis_dict, *count_average_size(analysis_dict, "patch"))
            analysis_dict = update_slic_analysis_dict(analysis_dict, *get_patch_sizes(patch_seg))
            cli_out.new_line()
            
            # update info to toml file
            toml_file = slic_dir.joinpath(f"{result_name}.ana.toml")
            update_toml_file(toml_file, analysis_dict)
            
            # update pbar
            pbar.advance(task)

    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/