import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import skimage as ski
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.dataset.dsname import get_dsname_sortinfo
from modules.data.dname import get_dname_sortinfo
from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.dataset.augmentation import crop_base_size
from modules.ml.calc_seg_feat import (count_area, count_average_size,
                                      count_element, get_patch_sizes,
                                      update_ana_toml_file,
                                      update_seg_analysis_dict)
from modules.ml.utils import get_seg_desc, get_slic_param_name, parse_base_size
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, get_repo_root

install()
# -----------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")

    """ Init components """
    cli_out = CLIOutput()
    cli_out.divide()
    path_navigator = PathNavigator()
    processed_di = ProcessedDataInstance()
    processed_di.parse_config("ml_analysis.toml")
    
    """ Load config """
    config = load_config("ml_analysis.toml")
    # [data_processed]
    palmskin_result_name: Path = Path(config["data_processed"]["palmskin_result_name"])
    cluster_desc: str = config["data_processed"]["cluster_desc"]
    # [seg_results]
    seg_desc = get_seg_desc(config)
    dataset_base_size = config["seg_results"]["base_size"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()
    
    # base size cropper
    base_size_cropper = crop_base_size(*parse_base_size(config))
    
    # get `seg_dirname`
    if seg_desc == "SLIC":
        seg_param_name = get_slic_param_name(config)
    elif seg_desc == "Cellpose":
        seg_param_name = "model_id" # TBD
    seg_dirname = f"{palmskin_result_name.stem}.{seg_param_name}"
    
    """ Colloct image file (dsname.tiff) """
    dataset_cropped: Path = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
    src_root = dataset_cropped.joinpath(cluster_desc.split("_")[-1], # e.g. RND2022
                                        processed_di.instance_name,
                                        palmskin_result_name.stem,
                                        dataset_base_size)
    ds_imgs = sorted(src_root.glob("*/*/*.tiff"), key=get_dsname_sortinfo)
    print(f"Total files: {len(ds_imgs)}")
    
    """ Processed Data Instance """
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
            ana_toml_file = ds_seg_dir.joinpath(f"{seg_dirname}.ana.toml")
            update_ana_toml_file(ana_toml_file, analysis_dict)
            
            # update pbar
            pbar.advance(task)

    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/