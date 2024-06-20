import json
import os
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich import print
from rich.console import Console
from rich.pretty import Pretty
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


def calc_darkratio_stat(dark_ratios: list[float]) -> tuple[float]:
    """_summary_

    Args:
        dark_ratios (list[float]): _description_

    Returns:
        tuple[float]: (`q2`, `mean`, `std`)
    """    
    n_sample = len(dark_ratios)
    q2 = np.quantile(dark_ratios, 0.5)
    mean = np.mean(dark_ratios)
    std = np.std(dark_ratios)
    print(f"Position: '{pos}'")
    print(f"- n_sample = {n_sample}")
    print(f"- Q2 `dark_ratio` = {round(q2, 3)}")
    print(f"- Avg `dark_ratio` = {mean} ± {std}")
    print(f"- (round) Avg `dark_ratio` = {round(mean, 3)} ± {round(std, 3)}\n")
    
    return q2, mean, std
    # -------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")
    
    # init components
    console = Console()
    cli_out = CLIOutput()
    path_navigator = PathNavigator()
    
    # maunal variables
    filter_dict: dict[str, str] = {}
    metrics_dict: dict[str, tuple] = {}
    
    
    """ Load config """
    cli_out.divide(title="Load config")
    
    # load `config`
    config = load_config("calc_avg_nocrop_darkratio.toml")
    # [model_prediction]
    model_time_stamp: str = config["model_prediction"]["time_stamp"]
    model_state: str = config["model_prediction"]["state"]
    # [col_filter]
    filter_dict["fish_pos"] = check_filter(config["col_filter"]["fish_pos"])
    filter_dict["dataset"] = check_filter(config["col_filter"]["dataset"])
    # history_dir
    history_dir = get_history_dir(path_navigator,
                                  model_time_stamp, model_state,
                                  cli_out)
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
    
    
    """ Calculate `thresholded_cam_area` on cell """
    cli_out.divide("Calculate average `dark_ratio`")
    
    for k, v in filter_dict.items():
        if v != "":
            dataset_df = dataset_df[(dataset_df[k] == v)]
    
    # copy full df for convenience to plot `A+P``
    if filter_dict["fish_pos"] == "":
        tmp_df = deepcopy(dataset_df)
        tmp_df["fish_pos"] = "A+P"
        dataset_df = pd.concat([dataset_df, tmp_df])
    
    # rename column
    dataset_df = dataset_df.rename(columns={"fish_pos": "position"})
    
    # calculate metrics, prepare `legend_labels`
    pos_order: list[str] = list(Counter(dataset_df["position"]).keys())
    legend_labels = []
    for pos in pos_order:
        dark_ratios = list(dataset_df[(dataset_df["position"] == pos)]["dark_ratio"])
        metrics_dict[pos] = calc_darkratio_stat(dark_ratios)
        legend_labels.append("{} (Q2: {:.3f}, Avg: {:.3f} ± {:.3f})".format(pos, *metrics_dict[pos]))
    
    # plot histogram with kde
    ax = sns.histplot(data=dataset_df, x="dark_ratio", hue="position",
                      kde=True, binwidth=0.05, stat="probability",
                      hue_order=pos_order, element="step",
                      common_norm=False)
    legend = ax.get_legend()
    handles = legend.legend_handles
    ax.legend(handles, legend_labels, fontsize=8)
    
    # set title
    desc: list[str] = []
    desc.extend(["DarkRatio", f"{dataset_base_size}"])
    if filter_dict["dataset"] != "": desc.append(f"{filter_dict['dataset']}")
    if filter_dict["fish_pos"] != "": desc.append(f"{filter_dict['fish_pos']}Only")
    ax.set_title(", ".join(desc))
    
    # save figure
    desc[0] = desc[0].lower()
    fig_path = src_root.joinpath(f"NoCropDarkRatio/{'_'.join(desc)}.png")
    create_new_dir(fig_path.parent)
    fig = ax.get_figure()
    fig.savefig(fig_path)
    print(f"saved figure: '{fig_path}'")
    
    # show figure (needs x11 forwarding)
    if "DISPLAY" in os.environ:
        plt.show()
    else:
        print("\n[orange1]Warning: Can't find `DISPLAY` in `os.environ`, plot will not show.")
    
    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/