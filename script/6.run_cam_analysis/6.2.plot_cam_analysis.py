import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rich.progress
import seaborn as sns
from rich import print
from rich.console import Console
from rich.pretty import Pretty
from rich.traceback import install
from scipy.stats import ttest_ind

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

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
    # [empty]
    
    
    """ Load config """
    cli_out.divide(title="Load config")
    
    # load `config`
    config = load_config("6.run_cam_analysis.toml")
    # [model_prediction]
    model_time_stamp: str = config["model_prediction"]["time_stamp"]
    model_state: str = config["model_prediction"]["state"]
    # [cam_analysis]
    cam_top_n_area = config["cam_analysis"]["top_n_area"]
    # history_dir
    history_dir = get_history_dir(path_navigator,
                                  model_time_stamp, model_state,
                                  cli_out)
    cam_result_root = history_dir.joinpath("cam_result")
    
    
    """ Collect `thresed_cam_area_on_cell` """
    cli_out.divide(title="Collect `thresed_cam_area_on_cell` for each subcrop image")
    
    # read file: '{Logs}_thresed_cam_area_on_cell.log'
    file_name = r"{Logs}_thresed_cam_area_on_cell.log"
    print(f"file : '{file_name}'")
    with rich.progress.open(cam_result_root.joinpath(file_name), mode="r") as f_reader:
        pred_ans_dict: dict[str, dict] = json.load(f_reader)
    cli_out.new_line()
    
    # get `prob_thres`
    probs = []
    for k, v in pred_ans_dict.items():
        probs.append(v["pred_prob"][v["pred"]])

    print("( min, q1, q2, q3, max ) = (",
            np.min(probs),
            np.quantile(probs, 0.25), np.quantile(probs, 0.5), np.quantile(probs, 0.75), 
            np.max(probs), ")\n")
    prob_thres = np.quantile(probs, 0.5)

    # collect `thresed_cam_area_on_cell` 
    thres_area_dict = {}
    thres_area_dict["S"] = []
    thres_area_dict["M"] = []
    thres_area_dict["L"] = []

    for k, v in pred_ans_dict.items():
        
        if v["pred"] == v["gt"]:
            if (v["pred_prob"][v["pred"]] > prob_thres):
                data_filtered = []
                try:
                    # method1 (area > q3)
                    # q3 = np.quantile(v["thresed_cam_area_on_cell"], 0.75)
                    # data_filtered = [x for x in v["thresed_cam_area_on_cell"] if x > q3]
                    
                    # method2 (top_n_area)
                    data_filtered = v["thresed_cam_area_on_cell"][:cam_top_n_area]
                    
                except IndexError:
                    pass
                thres_area_dict[v["pred"]].extend(data_filtered)
    
    
    """ Generate violin plot """
    cli_out.divide("Generate violin plot")
    
    thres_area_dict.pop("M")
    sns.violinplot(data=thres_area_dict)
    
    plt.title(f"Violinplot of different size, top {cam_top_n_area} area")
    plt.xlabel("Groups")
    plt.ylabel("Pixels")
    
    # p value
    t_stat, p_val = ttest_ind(thres_area_dict["S"], thres_area_dict["L"])
    print(f"T-statistic: {t_stat}, P-value: {p_val}")

    # set y-axis limit
    q1 = np.max([np.quantile(thres_area_dict['S'], 0.25),
                #  np.quantile(thres_area_dict['M'], 0.25),
                 np.quantile(thres_area_dict['L'], 0.25)])
    q3 = np.max([np.quantile(thres_area_dict['S'], 0.75),
                #  np.quantile(thres_area_dict['M'], 0.75),
                 np.quantile(thres_area_dict['L'], 0.75)])
    plt.ylim(q1 - 2 * (q3 - q1), q3 + 2 * (q3 - q1))

    # save fig
    fig_path = cam_result_root.joinpath(f"violinplot_top_{cam_top_n_area}.png")
    plt.savefig(fig_path)
    print(f"violin plot : '{fig_path.resolve()}'")
    

    print("(S, L) = ({},  {})".format(
        np.quantile(thres_area_dict['S'], 0.5),
        # np.quantile(thres_area_dict['M'], 0.5),
        np.quantile(thres_area_dict['L'], 0.5)))
    cli_out.new_line()
    
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/