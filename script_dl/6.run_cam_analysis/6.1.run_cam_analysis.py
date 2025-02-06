import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rich.progress
import skimage as ski
from rich import print
from rich.console import Console
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.dataset.utils import parse_dataset_file_name
from modules.dl.cam.analysis import calc_thresed_cam_area_on_cell
from modules.dl.tester.utils import get_history_dir
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import get_repo_root

install()
# -----------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")
    
    # init components
    console = Console()
    cli_out = CLIOutput()
    path_navigator = PathNavigator()
    
    # maunal variables
    cam_threshold = 120
    img_dict: dict[str, np.ndarray] = {}
    
    
    """ Load config """
    cli_out.divide(title="Load config")
    
    # load `config`
    config = load_config("6.run_cam_analysis.toml")
    # [model_prediction]
    model_time_stamp: str = config["model_prediction"]["time_stamp"]
    model_state: str = config["model_prediction"]["state"]
    # [cam_analysis]
    cam_threshold = config["cam_analysis"]["threshold"]
    # history_dir
    history_dir = get_history_dir(path_navigator,
                                  model_time_stamp, model_state,
                                  cli_out)
    cam_result_root = history_dir.joinpath("cam_result")
    
    # load `training_config`
    training_config = load_config(history_dir.joinpath("training_config.toml"))
    # [dataset]
    dataset_seed_dir: str = training_config["dataset"]["seed_dir"]
    dataset_data: str = training_config["dataset"]["data"]
    dataset_palmskin_result: str = training_config["dataset"]["palmskin_result"]
    # dataset_palmskin_result: str = "28_RGB_m3d"
    dataset_base_size: str = training_config["dataset"]["base_size"]
    dataset_classif_strategy: str = training_config["dataset"]["classif_strategy"]
    dataset_file_name: str = training_config["dataset"]["file_name"]
    intensity = parse_dataset_file_name(dataset_file_name)["intensity"]
    print(f"intensity = {intensity}")
    
    # dataset_df (finding original image)
    dataset_cropped: Path = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
    src_root = dataset_cropped.joinpath(dataset_seed_dir,
                                        dataset_data,
                                        dataset_palmskin_result,
                                        dataset_base_size)
    dataset_file: Path = src_root.joinpath(dataset_classif_strategy,
                                           dataset_file_name)
    dataset_df: pd.DataFrame = pd.read_csv(dataset_file, encoding='utf_8_sig')
    
    
    """ Calculate `thresholded_cam_area` on cell """
    cli_out.divide("Calculate `thresholded_cam_area` on cell")
    
    # read file: '{Logs}_PredByFish_predict_ans.log'
    file_name = r"{Logs}_PredByFish_predict_ans.log"
    print(f"file : '{file_name}'")
    with rich.progress.open(history_dir.joinpath(file_name), mode="r") as f_reader:
        pred_ans_dict: dict[str, dict] = json.load(f_reader)
    
    # calculate `cam_thres_on_cell` area
    cli_out.new_line()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(pred_ans_dict))
        
        for k, v in pred_ans_dict.items():
            
            pbar.update(task, description=f"[yellow]{k} : ")
            
            # read original image
            orig_path = Path(list(dataset_df[(dataset_df["image_name"] == k)]["path"])[0])
            img_dict["orig"] = ski.io.imread(src_root.joinpath(orig_path))
            
            # read cam image
            cam_name = k.replace("crop", "graymap")
            cam_path_list = list(cam_result_root.glob(f"*/grayscale_map/{cam_name}.tiff"))
            assert len(cam_path_list) == 1, f"CAM of {cam_name} is not unique."
            img_dict["cam"] = ski.io.imread(cam_path_list[0])
            
            tmp_list = calc_thresed_cam_area_on_cell(img_dict["orig"], intensity,
                                                     img_dict["cam"], cam_threshold)
            pred_ans_dict[k]["thresed_cam_area_on_cell"] = sorted(tmp_list, reverse=True)
            
            # update pbar
            pbar.advance(task)
    cli_out.new_line()
    
    # write file: '{Logs}_thresed_cam_area_on_cell.log'
    with console.status("Writing..."):
        
        file_name = r"{Logs}_thresed_cam_area_on_cell.log"
        print(f"output file : '{file_name}'")
        with open(cam_result_root.joinpath(file_name), mode="w") as f_writer:
            json.dump(pred_ans_dict, f_writer, indent=4)
    
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/