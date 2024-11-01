import os
import pickle
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import skimage as ski
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.dataset.dsname import get_dsname_sortinfo
from modules.data.dname import get_dname_sortinfo
from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.dataset.augmentation import crop_base_size
from modules.ml.utils import get_slic_param_name
from modules.shared.clioutput import CLIOutput
from modules.shared.config import dump_config, load_config
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


def update_seg_analysis_dict(analysis_dict:dict[str, Any], value:Any, key:str) -> dict[str, Any]:
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
    cli_out = CLIOutput()
    cli_out.divide()
    path_navigator = PathNavigator()
    processed_di = ProcessedDataInstance()
    
    """ Load config """
    # `dark` and `merge` are two parameters as color space distance, determined by experiences
    config = load_config("7.a.1.crop_seg_results.toml")
    # 
    seg_desc = config["seg_desc"]
    accept_str = ["SLIC", "Cellpose"]
    if seg_desc not in accept_str:
        raise ValueError(f"`config.seg_desc`, only accept {accept_str}\n")
    # [dataset]
    dataset_seed_dir: str = config["dataset"]["seed_dir"]
    dataset_data: str = config["dataset"]["data"]
    dataset_palmskin_result: str = config["dataset"]["palmskin_result"]
    dataset_base_size: str = config["dataset"]["base_size"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()
    
    # base size cropper
    size: list[str] = dataset_base_size.split("_")
    size_w = int(size[0].replace("W", ""))
    assert size_w <= 512, "Maximum support `width` is 512"
    size_h = int(size[1].replace("H", ""))
    assert size_h <= 1024, "Maximum support `height` is 1024"
    base_size_cropper = crop_base_size(size_w, size_h)
    
    # get `seg_dirname`
    if seg_desc == "SLIC":
        seg_param_name = get_slic_param_name(config)
    elif seg_desc == "Cellpose":
        seg_param_name = "model_id" # TBD
    seg_dirname = f"{dataset_palmskin_result}.{seg_param_name}"
    
    """ Colloct image file (dsname.tiff) """
    dataset_cropped: Path = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
    src_root = dataset_cropped.joinpath(dataset_seed_dir,
                                        dataset_data,
                                        dataset_palmskin_result,
                                        dataset_base_size)
    ds_imgs = sorted(src_root.glob("*/*/*.tiff"), key=get_dsname_sortinfo)
    print(f"Total files: {len(ds_imgs)}")
    
    """ Processed Data Instance """
    instance_desc = re.split("{|}", dataset_data)[1]
    temp_dict = {"data_processed": {"instance_desc": instance_desc}}
    processed_di.parse_config(temp_dict)
    csv_path = processed_di.instance_root.joinpath("data.csv")
    df: pd.DataFrame = pd.read_csv(csv_path, encoding='utf_8_sig')
    palmskin_dnames = sorted(pd.concat([df["Palmskin Anterior (SP8)"],
                                        df["Palmskin Posterior (SP8)"]]),
                            key=get_dname_sortinfo)

    """ Main Process: Crop Segment Results """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(ds_imgs))
        
        for ds_img, dname in zip(ds_imgs, palmskin_dnames):
            
            # check dsname, dname is match
            assert get_dsname_sortinfo(ds_img) == get_dname_sortinfo(dname), "dsname, dname not match"
            
            # dname_dir
            dname_dir = processed_di.palmskin_processed_dname_dirs_dict[dname]
            d_seg_dir = dname_dir.joinpath(f"{seg_desc}/{seg_dirname}")
            
            # dsname_dir
            dsname_dir = ds_img.parent
            ds_seg_dir = dsname_dir.joinpath(f"{seg_desc}/{seg_dirname}")
            create_new_dir(ds_seg_dir)
            print(f"[ {dsname_dir.stem} : '{ds_seg_dir}' ]")
            
            # crop `png` files
            for d_path in d_seg_dir.glob("*.png"):
                png = base_size_cropper(image=ski.io.imread(d_path))
                save_path = ds_seg_dir.joinpath(f"{d_path.name}")
                ski.io.imsave(save_path, png)
            
            # crop `pkl` files
            cell_seg: np.ndarray = None
            patch_seg: np.ndarray = None
            for d_path in d_seg_dir.glob("*.pkl"):
                
                # load `pkl`
                with open(d_path, mode="rb") as f_reader:
                    seg = pickle.load(f_reader)
                assert isinstance(seg, np.ndarray), \
                    f"Warning '{d_path.stem}' might be replaced by others"
                
                # cropping
                seg = base_size_cropper(image=seg)
                
                # dump `pkl`
                save_path = ds_seg_dir.joinpath(f"{d_path.name}")
                with open(save_path, mode="wb") as f_writer:
                    pickle.dump(seg, f_writer)
                
                if d_path.stem.endswith("seg1"):
                    cell_seg = deepcopy(seg)
                if d_path.stem.endswith("seg2"):
                    patch_seg = deepcopy(seg)
            
            assert isinstance(cell_seg, np.ndarray)
            assert isinstance(patch_seg, np.ndarray)
            
            # update
            analysis_dict = {}
            analysis_dict = update_seg_analysis_dict(analysis_dict, *count_area(cell_seg))
            analysis_dict = update_seg_analysis_dict(analysis_dict, *count_element(cell_seg, "cell"))
            analysis_dict = update_seg_analysis_dict(analysis_dict, *count_element(patch_seg, "patch"))
            analysis_dict = update_seg_analysis_dict(analysis_dict, *count_average_size(analysis_dict, "cell"))
            analysis_dict = update_seg_analysis_dict(analysis_dict, *count_average_size(analysis_dict, "patch"))
            analysis_dict = update_seg_analysis_dict(analysis_dict, *get_patch_sizes(patch_seg))
            cli_out.new_line()
            
            # update info to toml file
            toml_file = ds_seg_dir.joinpath(f"{seg_dirname}.ana.toml")
            update_toml_file(toml_file, analysis_dict)
            
            # update pbar
            pbar.advance(task)

    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/