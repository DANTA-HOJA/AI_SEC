import json
import os
import re
import sys
from pathlib import Path

import cv2
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

from modules.data.dname import get_dname_sortinfo
from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.tester.utils import get_history_dir
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, get_repo_root

install()
# -----------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")
    
    """ Init components """
    console = Console()
    cli_out = CLIOutput()
    processed_di = ProcessedDataInstance()
    path_navigator = PathNavigator()
    
    """ Load `config` """
    cli_out.divide(title="Load config")
    config = load_config("5.make_cam_gallery.toml")
    # [model_prediction]
    model_time_stamp: str = config["model_prediction"]["time_stamp"]
    model_state: str = config["model_prediction"]["state"]
    # [draw.cam_image]
    cam_weight: float = config["draw"]["cam_image"]["weight"]
    if not cam_weight: cam_weight = 0.5
    
    """ Set `history_dir` """
    history_dir = get_history_dir(path_navigator,
                                  model_time_stamp, model_state,
                                  cli_out)
    cam_result_root = history_dir.joinpath("cam_result")
    cam_gallery_dir = history_dir.joinpath("+---CAM_Gallery")
    if cam_gallery_dir.exists():
        raise FileExistsError(f"Directory already exists: '{cam_gallery_dir}'. "
                              f"To re-generate, please delete it manually.")
    console.line()
    console.print("[yellow]config: \n", Pretty(config))
    
    """ Load `training_config` """
    training_config = load_config(history_dir.joinpath("training_config.toml"))
    # [dataset]
    dataset_seed_dir: str = training_config["dataset"]["seed_dir"]
    dataset_data: str = training_config["dataset"]["data"]
    # dataset_palmskin_result: str = training_config["dataset"]["palmskin_result"]
    # dataset_base_size: str = training_config["dataset"]["base_size"]
    dataset_classif_strategy: str = training_config["dataset"]["classif_strategy"]
    dataset_file_name: str = training_config["dataset"]["file_name"]
    assert dataset_file_name == "DS_SURF3C_NORMBF.xxx", \
        f"DatasetFile must be 'DS_SURF3C_NORMBF.xxx', current: '{dataset_file_name}'"
    # get `cluster_desc`
    tmp_list = [dataset_file_name.split("_")[1],
                dataset_classif_strategy,
                dataset_seed_dir]
    cluster_desc: str = f"{'_'.join(tmp_list)}" # e.g. 'SURF3C_KMeansORIG_RND2022'
    console.line()
    console.print("[yellow]training_config.note: \n", Pretty(training_config["note"], expand_all=True), "\n")
    console.print("[yellow]training_config.dataset: \n", Pretty(training_config["dataset"], expand_all=True), "\n")
    
    # set config to `processed_di` for finding original image
    instance_desc = re.split("{|}", dataset_data)[1]
    temp_dict = {"data_processed": {"instance_desc": instance_desc}}
    processed_di.parse_config(temp_dict)
    
    """ Create NormBF CAM gallery """
    cli_out.divide("Create NormBF CAM Gallery")
    
    # glob `dnames` in `cam_result` dir
    dnames = sorted(cam_result_root.glob("*"), key=get_dname_sortinfo)
    dnames = [dname.stem for dname in dnames]
    
    # get median image size
    file_name = "median_image_size.toml"
    console.print(f"Read file : '{file_name}'")
    tmp_path = history_dir.joinpath(file_name)
    median_image_size = tuple(load_config(tmp_path)["image_size"])
    
    # read file: '{Logs}_PredByFish_predict_ans.log'
    file_name = r"{Logs}_PredByFish_predict_ans.log"
    console.print(f"Read file : '{file_name}'")
    tmp_path = history_dir.joinpath(file_name)
    with rich.progress.open(tmp_path, mode="r") as f_reader:
        pred_ans_dict: dict[str, dict] = json.load(f_reader)
    console.line()
    
    # create nocrop cam gallery
    with Progress(SpinnerColumn(),
                  *Progress.get_default_columns(),
                  TextColumn("{task.completed} of {task.total}"),
                  auto_refresh=False,
                  console=console) as pbar:
        
        task = pbar.add_task("[cyan]Processing...", total=len(dnames))
        
        # create figure
        dpi = 100
        figsize = np.array(median_image_size)/dpi
        figsize[1] = figsize[1]*1.35
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        
        for dname in dnames:
            
            pbar.update(task, description=f"[yellow]{dname} : ")
            pbar.refresh()
            
            # img: original
            orig_path = processed_di.brightfield_processed_dname_dirs_dict[dname]
            orig_img = cv2.imread(str(orig_path.joinpath("Norm_BF.tif")))
            
            # img: cam
            cam_path = cam_result_root.joinpath(f"{dname}/color_map/{dname}.colormap.tiff")
            try:
                cam_img = ski.io.imread(cam_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                        f"Can't find CAM image '{dname}.colormap.tiff' in `cam_result` dir, "
                        f"run one of script in '4.test_by_fish/*.py' to create CAM.")
            
            # img: cam on original
            overlay_img = (orig_img/255)*cam_weight + (cam_img/255)*cam_weight
            overlay_img = np.uint8(overlay_img*255)
            
            # get predict info ('{Logs}_PredByFish_predict_ans.log')
            gt_cls: str = pred_ans_dict[dname]["gt"]
            pred_cls: str = pred_ans_dict[dname]["pred"]
            pred_prob: dict[float] = pred_ans_dict[dname]["pred_prob"]
            gt_pred_prob: float = pred_ans_dict[dname]["pred_prob"][gt_cls]
            
            # plot
            fig.suptitle(f"[{gt_cls}] {dname} : {dataset_data}, {cluster_desc}")
            ax.set_axis_off()
            
            ax.imshow(orig_img)
            title = f"median_image_size : {median_image_size}"
            ax.set_title(title, fontsize=10)
            fig.tight_layout()
            fig_path = cam_gallery_dir.joinpath(f"{gt_cls}/{gt_pred_prob:0.5f}_{dname}_orig.png")
            create_new_dir(fig_path.parent)
            fig.savefig(fig_path)
            console.print(f"[{gt_cls}] ({gt_pred_prob:0.5f}) {dname}_orig : '{fig_path}'")
            
            ax.imshow(overlay_img)
            title = "predict : correct" if gt_cls == pred_cls else f"predict : {pred_cls}"
            title += f", {json.dumps(pred_prob)}"
            ax.set_title(title, fontsize=10)
            # fig.tight_layout()
            fig_path = cam_gallery_dir.joinpath(f"{gt_cls}/{gt_pred_prob:0.5f}_{dname}_overlay.png")
            fig.savefig(fig_path)
            console.print(f"[{gt_cls}] ({gt_pred_prob:0.5f}) {dname}_overlay : '{fig_path}'")
            
            ax.clear()
            console.line()
            pbar.update(task, advance=1)
            pbar.refresh()
    
    console.line()
    console.print("[green]Done! \n")
    # -------------------------------------------------------------------------/