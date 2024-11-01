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
from modules.ml.seg_genertor import single_slic_labeling
from modules.ml.utils import get_slic_param_name
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.utils import create_new_dir, get_repo_root

install()
# -----------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")

    """ Init components """
    cli_out = CLIOutput()
    cli_out.divide()
    processed_di = ProcessedDataInstance()
    processed_di.parse_config("ml_analysis.toml")
    w512h1024_cropper = crop_base_size(512, 1024)

    # load config
    # `dark` and `merge` are two parameters as color space distance, determined by experiences
    config = load_config("ml_analysis.toml")
    palmskin_result_name: Path = Path(config["data_processed"]["palmskin_result_name"])
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    merge: int       = config["SLIC"]["merge"]
    debug_mode: bool = config["SLIC"]["debug_mode"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()

    # get `slic_dirname`
    slic_param_name = get_slic_param_name(config)
    slic_dirname = f"{palmskin_result_name.stem}.{slic_param_name}"
    
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
            
            # create `slic_dir`
            slic_dir = dname_dir.joinpath(f"SLIC/{slic_dirname}")
            create_new_dir(slic_dir)
            
            # get image, size: W512_H1024 (FixedROI)
            target_path = slic_dir.parent.joinpath(f"{result_path.stem}.W512_H1024.tif")
            if not target_path.exists():
                tmp_img = w512h1024_cropper(image=cv2.imread(str(result_path)))
                cv2.imwrite(str(target_path), tmp_img)
            
            # slic
            cell_seg, patch_seg = single_slic_labeling(slic_dir, target_path,
                                                       n_segments, dark, merge,
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
            toml_file = slic_dir.joinpath(f"{slic_dirname}.ana.toml")
            update_ana_toml_file(toml_file, analysis_dict)
            
            # update pbar
            pbar.advance(task)

    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/