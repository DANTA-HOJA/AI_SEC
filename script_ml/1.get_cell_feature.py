import os
import sys
from pathlib import Path

import cv2
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.dataset.augmentation import crop_base_size
from modules.ml.calc_seg_feat import (count_area, count_average_size,
                                      count_element, get_patch_sizes,
                                      update_ana_toml_file,
                                      update_seg_analysis_dict)
from modules.ml.seg_generate import (single_cellpose_prediction,
                                     single_slic_labeling)
from modules.ml.utils import (get_cellpose_param_name, get_seg_desc,
                              get_slic_param_name)
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
    # [seg_results]
    seg_desc = get_seg_desc(config)
    # [SLIC]
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    # [Cellpose]
    cp_model_name: str = config["Cellpose"]["cp_model_name"]
    channels: list     = config["Cellpose"]["channels"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()

    # 'W512_H1024' cropper
    w512h1024_cropper = crop_base_size(512, 1024)
    
    # get `seg_dirname`
    merge: int       = config[f"{seg_desc}"]["merge"]
    debug_mode: bool = config[f"{seg_desc}"]["debug_mode"]
    if seg_desc == "SLIC":
        seg_param_name = get_slic_param_name(config)
    elif seg_desc == "Cellpose":
        # check model
        cp_model_dir = path_navigator.dbpp.get_one_of_dbpp_roots("model_cellpose")
        cp_model_path = cp_model_dir.joinpath(cp_model_name)
        if cp_model_path.is_file():
            seg_param_name = get_cellpose_param_name(config)
        else:
            raise FileNotFoundError(f"'{cp_model_path}' is not a file or does not exist")
        # load model
        if "cellpose" in sys.executable: # check python environment
                from cellpose import models as cpmodels
                cp_model = cpmodels.CellposeModel(gpu=True, pretrained_model=str(cp_model_path))
        else:
            raise RuntimeError("Detect environment name not for Cellpose. "
                                "Please follow the setup instructions provided at "
                                "'https://github.com/MouseLand/cellpose' "
                                "to create an environment.")
    seg_dirname = f"{palmskin_result_name.stem}.{seg_param_name}"
    
    """ Colloct image file names """
    rel_path, sorted_results_dict = \
        processed_di.get_sorted_results_dict("palmskin", str(palmskin_result_name))
    result_paths = list(sorted_results_dict.values())
    print(f"Total files: {len(result_paths)}")

    """ Apply SLIC on each image """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(result_paths))
        
        for result_path in result_paths:
            
            dname_dir = Path(str(result_path).replace(rel_path, ""))
            print(f"[ {dname_dir.parts[-1]} ]")
            
            # get image, size: W512_H1024 (FixedROI)
            target_path = dname_dir.joinpath(f"CenterCropped/{result_path.stem}.W512_H1024.tif")
            if not target_path.exists():
                create_new_dir(target_path.parent)
                tmp_img = w512h1024_cropper(image=cv2.imread(str(result_path)))
                cv2.imwrite(str(target_path), tmp_img)
            
            # dname_dir
            d_seg_dir = dname_dir.joinpath(f"{seg_desc}/{seg_dirname}")
            create_new_dir(d_seg_dir)
            
            # generate cell segmentation
            if seg_desc == "SLIC":
                cell_seg, patch_seg = single_slic_labeling(d_seg_dir, target_path,
                                                           n_segments, dark, merge,
                                                           debug_mode)
            elif seg_desc == "Cellpose":
                cell_seg, patch_seg = single_cellpose_prediction(d_seg_dir, target_path,
                                                                 channels, cp_model, merge,
                                                                 debug_mode)
            
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
            ana_toml_file = d_seg_dir.joinpath(f"{seg_dirname}.ana.toml")
            update_ana_toml_file(ana_toml_file, analysis_dict)
            
            # update pbar
            pbar.advance(task)

    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/