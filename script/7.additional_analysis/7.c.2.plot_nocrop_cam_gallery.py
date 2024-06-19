import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich.progress
import skimage as ski
from rich import print
from rich.console import Console
from rich.pretty import Pretty
from rich.progress import *
from rich.traceback import install

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.dl.tester.utils import get_history_dir
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, get_repo_root

install()
# -----------------------------------------------------------------------------/


def check_filter(keywords: list[str]):
    """
    """
    if len(keywords) == 0:
        return ""
    elif len(keywords) == 1:
        return keywords[0]
    else:
        raise ValueError(f"Too many filter: {keywords}")
    # -------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")
    
    # init components
    console = Console()
    cli_out = CLIOutput()
    path_navigator = PathNavigator()
    
    # maunal variables
    filter_dict = {}
    
    
    """ Load config """
    cli_out.divide(title="Load config")
    
    # load `config`
    config = load_config("calc_avg_nocrop_darkratio.toml")
    # [model_prediction]
    model_time_stamp: str = config["model_prediction"]["time_stamp"]
    model_state: str = config["model_prediction"]["state"]
    # [col_filter]
    filter_dict["fish_pos"] = check_filter(config["col_filter"]["fish_pos"])
    filter_dict["dataset"] = "test"
    # history_dir
    history_dir = get_history_dir(path_navigator,
                                  model_time_stamp, model_state,
                                  cli_out)
    cam_result_root = history_dir.joinpath("cam_result")
    cam_gallery_dir = history_dir.joinpath("+---CAM_Gallery")
    if cam_gallery_dir.exists():
        raise FileExistsError(f"Directory already exists: '{cam_gallery_dir}'. "
                              f"To re-generate, please delete it manually.")
    cli_out.new_line()
    print("[yellow]config: \n", Pretty(config, expand_all=True))
    
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
    assert dataset_file_name == "DS_SURF3C_NOCROP.csv", \
        f"DatasetFile must be 'DS_SURF3C_NOCROP.csv', current: {dataset_file_name}"
    cli_out.new_line()
    print("[yellow]training_config.note: \n", Pretty(training_config["note"], expand_all=True))
    print("[yellow]training_config.dataset: \n", Pretty(training_config["dataset"], expand_all=True))
    
    # dataset_df (finding original image)
    dataset_cropped: Path = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
    src_root = dataset_cropped.joinpath(dataset_seed_dir,
                                        dataset_data,
                                        dataset_palmskin_result,
                                        dataset_base_size)
    dataset_file: Path = src_root.joinpath(dataset_classif_strategy,
                                           dataset_file_name)
    dataset_df: pd.DataFrame = pd.read_csv(dataset_file, encoding='utf_8_sig')
    
    
    """ Create nocrop cam gallery """
    cli_out.divide("Create NoCrop CAM Gallery")
    
    # filtered `dataset_df`
    for k, v in filter_dict.items():
        if v != "":
            dataset_df = dataset_df[(dataset_df[k] == v)]
    
    # read file: '{Logs}_PredByFish_predict_ans.log'
    file_name = r"{Logs}_PredByFish_predict_ans.log"
    print(f"file : '{file_name}'")
    with rich.progress.open(history_dir.joinpath(file_name), mode="r") as f_reader:
        pred_ans_dict: dict[str, dict] = json.load(f_reader)
    cli_out.new_line()
    
    # create nocrop cam gallery
    with Progress(SpinnerColumn(),
                  *Progress.get_default_columns(),
                  TextColumn("{task.completed} of {task.total}"),
                  auto_refresh=False) as pbar:
        
        task = pbar.add_task("[cyan]Processing...", total=len(dataset_df))
        
        # create figure
        fig, ax = plt.subplots(1, 1, figsize=(5.12, 10.24), dpi=100)
        
        for img_name in dataset_df["image_name"]:
            
            pbar.update(task, description=f"[yellow]{img_name} : ")
            pbar.refresh()
            
            # filter row
            target_row = dataset_df[(dataset_df["image_name"] == img_name)]
            
            # img: original
            orig_path = src_root.joinpath(list(target_row["path"])[0])
            orig_img = ski.io.imread(orig_path)
            
            # img: cam
            cam_path = cam_result_root.joinpath(f"{img_name}/color_map/{img_name}_colormap.tiff")
            try:
                cam_img = ski.io.imread(cam_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                        f"Can't find CAM image '{img_name}_colormap.tiff' in `cam_result` dir, "
                        f"run one of script in '4.test_by_fish/*.py' to create CAM, "
                        f"and check that if the `history_dir` (id: '{model_time_stamp}') is a Aonly/Ponly training case")
            
            # img: cam on original
            overlay_img = (orig_img/255)*0.5 + (cam_img/255)*0.5
            overlay_img = np.uint8(overlay_img*255)
            
            # get predict info ('{Logs}_PredByFish_predict_ans.log')
            gt_cls: str = pred_ans_dict[img_name]["gt"]
            pred_cls: str = pred_ans_dict[img_name]["pred"]
            pred_prob: dict[float] = pred_ans_dict[img_name]["pred_prob"]
            gt_pred_prob: float = pred_ans_dict[img_name]["pred_prob"][gt_cls]
            
            # get `dark_ratio` ('DS_SURF3C_NOCROP.csv')
            dark_ratio = float(list(target_row["dark_ratio"])[0])
            
            # plot
            fig.suptitle(f"[{gt_cls}] {img_name} : {dataset_palmskin_result}, {dataset_base_size}")
            # ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_axis_off()
            
            ax.imshow(orig_img)
            title = f"dark_ratio : {dark_ratio:0.3f}"
            ax.set_title(title, fontsize=10)
            fig.tight_layout()
            fig_path = cam_gallery_dir.joinpath(f"{gt_cls}/{gt_pred_prob:0.5f}_{img_name}_orig.png")
            create_new_dir(fig_path.parent)
            fig.savefig(fig_path)
            print(f"[{gt_cls}] ({gt_pred_prob:0.5f}) {img_name}_orig : '{fig_path}'")
            
            ax.imshow(overlay_img)
            title = "predict : correct" if gt_cls == pred_cls else f"predict : {pred_cls}"
            title += f", {json.dumps(pred_prob)}"
            ax.set_title(title, fontsize=10)
            fig.tight_layout()
            fig_path = cam_gallery_dir.joinpath(f"{gt_cls}/{gt_pred_prob:0.5f}_{img_name}_overlay.png")
            fig.savefig(fig_path)
            print(f"[{gt_cls}] ({gt_pred_prob:0.5f}) {img_name}_overlay : '{fig_path}'")
            
            ax.clear()
            cli_out.new_line()
            pbar.update(task, advance=1)
            pbar.refresh()
    
    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/