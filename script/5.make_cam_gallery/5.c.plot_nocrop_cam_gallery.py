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

from modules.data.dataset.dsname import get_dsname_sortinfo
from modules.dl.tester.utils import get_history_dir
from modules.plot.utils import add_detail_info, plt_to_pillow, pt_to_px
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
    dataset_palmskin_result: str = training_config["dataset"]["palmskin_result"]
    dataset_base_size: str = training_config["dataset"]["base_size"]
    dataset_classif_strategy: str = training_config["dataset"]["classif_strategy"]
    dataset_file_name: str = training_config["dataset"]["file_name"]
    assert dataset_file_name == "DS_SURF3C_NOCROP.csv", \
        f"DatasetFile must be 'DS_SURF3C_NOCROP.csv', current: '{dataset_file_name}'"
    console.line()
    console.print("[yellow]training_config.note: \n", Pretty(training_config["note"], expand_all=True), "\n")
    console.print("[yellow]training_config.dataset: \n", Pretty(training_config["dataset"], expand_all=True), "\n")
    
    # load `dataset_df` for finding original image
    dataset_cropped: Path = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
    src_root = dataset_cropped.joinpath(dataset_seed_dir,
                                        dataset_data,
                                        dataset_palmskin_result,
                                        dataset_base_size)
    dataset_file: Path = src_root.joinpath(dataset_classif_strategy,
                                           dataset_file_name)
    dataset_df: pd.DataFrame = pd.read_csv(dataset_file, encoding='utf_8_sig')
    
    
    """ Create Nocrop CAM gallery """
    cli_out.divide("Create NoCrop CAM Gallery")
    
    # glob `dsnames` in `cam_result` dir
    dsnames = sorted(cam_result_root.glob("*"), key=get_dsname_sortinfo)
    dsnames = [dsname.stem for dsname in dsnames]
    
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
        
        task = pbar.add_task("[cyan]Processing...", total=len(dsnames))
        
        # create figure
        dpi = 100
        figsize = np.array((512, 1024))/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        # font size
        font_pt = 12.0 # unit: pt
        font_px = pt_to_px(font_pt, dpi=dpi) # unit: pixel
        
        for dsname in dsnames:
            
            pbar.update(task, description=f"[yellow]{dsname} : ")
            pbar.refresh()
            
            # img: original
            target_row = dataset_df[(dataset_df["image_name"] == dsname)]
            orig_path = src_root.joinpath(list(target_row["path"])[0])
            orig_img = ski.io.imread(orig_path)
            
            # img: cam
            cam_path = cam_result_root.joinpath(f"{dsname}/color_map/{dsname}_colormap.tiff")
            try:
                cam_img = ski.io.imread(cam_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                        f"Can't find CAM image '{dsname}_colormap.tiff' in `cam_result` dir, "
                        f"run one of script in '4.test_by_fish/*.py' to create CAM.")
            
            # img: cam on original
            orig_img = (orig_img/255.0)
            cam_img = (cam_img/255.0)
            gamma_orig_img = np.power(orig_img, 0.5)
            overlay_img = cam_img*cam_weight + gamma_orig_img*(1 - cam_weight)
            
            # get predict info ('{Logs}_PredByFish_predict_ans.log')
            gt_cls: str = pred_ans_dict[dsname]["gt"]
            pred_cls: str = pred_ans_dict[dsname]["pred"]
            pred_prob: dict[float] = pred_ans_dict[dsname]["pred_prob"]
            gt_pred_prob: float = pred_ans_dict[dsname]["pred_prob"][gt_cls]
            
            # get `dark_ratio` ('DS_SURF3C_NOCROP.csv')
            dark_ratio = float(list(target_row["dark_ratio"])[0])
            
            # content
            content = []
            # `dark_ratio`
            content.extend(["➣ ", f"dark_ratio   : {dark_ratio:0.2%}", "\n"*2])
            # `{Logs}_PredByFish_predict_ans.log`
            content.extend(["➣ ", f'ground truth : "{gt_cls}"', "\n"*1])
            if gt_cls == pred_cls:
                content.extend(["➣ ", f'predict      : "correct"', "\n"*1])
            else:
                content.extend(["➣ ", f'predict      : "{pred_cls}"', "\n"*1])
            content.extend(["➣ ", "predicted probability : ", json.dumps(pred_prob), "\n"*2])
            # `training_config.toml`
            content.extend(["➣ ", "training_config.note :", "\n"*1])
            # `{Report}_PredByFish.log`
            content.extend([training_config["note"], "\n"*2])
            with rich.progress.open(history_dir.joinpath(r"{Report}_PredByFish.log"),
                                    mode="r", transient=True) as f_reader:
                content.extend([f_reader.read()])
            content = "".join(content)
            
            # plot
            ax.imshow(gamma_orig_img, vmin=0.0, vmax=1.0)
            ax.set_axis_off()
            ax.set_title(f"[{gt_cls}] {dsname}", fontsize=font_pt)
            fig.tight_layout()
            rgba_image = plt_to_pillow(fig)
            rgba_image = add_detail_info(rgba_image, content, font_size=font_px)
            fig_path = cam_gallery_dir.joinpath(f"{gt_cls}/{gt_pred_prob:0.5f}_{dsname}_orig.png")
            create_new_dir(fig_path.parent)
            rgba_image.save(fig_path)
            console.print(f"[{gt_cls}] ({gt_pred_prob:0.5f}) {dsname}_orig : '{fig_path}'")
            
            ax.imshow(overlay_img, vmin=0.0, vmax=1.0)
            ax.set_title(f"[{gt_cls}] {dsname}", fontsize=font_pt)
            rgba_image = plt_to_pillow(fig)
            rgba_image = add_detail_info(rgba_image, content, font_size=font_px)
            fig_path = cam_gallery_dir.joinpath(f"{gt_cls}/{gt_pred_prob:0.5f}_{dsname}_overlay.png")
            rgba_image.save(fig_path)
            console.print(f"[{gt_cls}] ({gt_pred_prob:0.5f}) {dsname}_overlay : '{fig_path}'")
            ax.clear()
            console.line()
            pbar.update(task, advance=1)
            pbar.refresh()
    
    console.line()
    console.print("[green]Done! \n")
    # -------------------------------------------------------------------------/