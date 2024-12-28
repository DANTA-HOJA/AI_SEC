import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
from modules.ml.utils import (get_cellpose_param_name, get_seg_desc,
                              get_slic_param_name)
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, get_repo_root

new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)

install()
# -----------------------------------------------------------------------------/


def get_size_sortinfo(size_dir: Path):
    """
    """
    size = size_dir.stem
    size: list[str] = size.split("_")
    size_w = int(size[0].replace("W", ""))
    size_h = int(size[1].replace("H", ""))
    
    return size_h, size_w
    # -------------------------------------------------------------------------/


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
    # [Cellpose]
    cp_model_name: str = config["Cellpose"]["cp_model_name"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()

    # get `seg_dirname`
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
    seg_dirname = f"{palmskin_result_name.stem}.{seg_param_name}"
    
    """ Processed Data Instance """
    csv_path = processed_di.instance_root.joinpath("data.csv")
    df: pd.DataFrame = pd.read_csv(csv_path, encoding='utf_8_sig')
    palmskin_dnames = sorted(pd.concat([df["Palmskin Anterior (SP8)"],
                                        df["Palmskin Posterior (SP8)"]]),
                            key=get_dname_sortinfo)
    
    """ Scan `size_dir` (**/W[]_H[]/) """
    dataset_cropped: Path = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
    src_root = dataset_cropped.joinpath(cluster_desc.split("_")[-1], # e.g. RND2022
                                        processed_di.instance_name,
                                        palmskin_result_name.stem)
    size_dirs = sorted(src_root.glob("*"), key=get_size_sortinfo, reverse=True)
    size_order: dict[str, str] = \
        {f"{size_dir.stem}": f"{size_dir.stem}" for size_dir in size_dirs }
    
    # new dataframe to store all `ana_toml_file`
    df = pd.DataFrame({'base_size': [], 'cell_number': []})
    
    """ Read analysis file """
    cli_out.divide()
    with Progress(transient=True) as pbar:
        task = pbar.add_task("[cyan]Size...", total=len(size_dirs))
        
        for size_dir in size_dirs:
            # update pbar display
            refresh_desc = f"[cyan]Size: '{size_dir.stem}'..."
            pbar.update(task, description=refresh_desc)
            cli_out.divide()
            
            # scan `ana_toml_file` under `size_dir`
            ds_ana_toml_files = list(size_dir.glob(f"**/{seg_dirname}.ana.toml"))
            print(f"Size: '{size_dir.stem}', Total files: {len(ds_ana_toml_files)}")
            
            # check if file missing
            if len(ds_ana_toml_files) != len(palmskin_dnames):
                size_order.pop(size_dir.stem)
                print(f"Size: '{size_dir.stem}' is skipped, "
                      f"expected number of images is {len(palmskin_dnames)}, "
                      f"but only {len(ds_ana_toml_files)} images are found.")
                continue
            
            task2 = pbar.add_task("[magenta]Seg_results...", total=len(ds_ana_toml_files))
            for ds_ana_toml_file in ds_ana_toml_files:
                # update pbar display
                dsname_dir = ds_ana_toml_file.parents[2]
                refresh_desc = f"[magenta]{seg_desc}, {seg_dirname}: '{dsname_dir.stem}'..."
                pbar.update(task2, description=refresh_desc)
                
                # load `ana_toml_file`
                analysis_dict = load_config(ds_ana_toml_file)
                tmp_df = pd.DataFrame({'base_size': [size_dir.stem],
                                       'cell_number': [analysis_dict["cell_count"]]})
                df = pd.concat([df, tmp_df], ignore_index=True)
                
                # update pbar
                pbar.advance(task2) # file
            
            # update pbar
            pbar.advance(task) # size
    
    
    """ Calculate statistics values """
    q1s = df.groupby(['base_size'])['cell_number'].quantile(0.25)
    q2s = df.groupby(['base_size'])['cell_number'].quantile(0.50)
    q3s = df.groupby(['base_size'])['cell_number'].quantile(0.75)
    means = df.groupby(['base_size'])['cell_number'].mean()
    stds = df.groupby(['base_size'])['cell_number'].std()
    
    """ Plot figure """
    fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=200)
    plt.title(seg_dirname, fontdict={"fontsize": 16})
    
    # legend
    legend_labels = []
    for size in size_order.keys():
        legend_labels.append("{} (Q2: {:.2f}, Avg: {:.2f} ± {:.2f})".format(size, q2s[size], means[size], stds[size]))
    
    # custom style for outlier
    # flierprops = dict(marker='o', markersize=5, markerfacecolor='red')
    
    # boxplot
    ax = sns.boxplot(
        ax=ax,
        x='base_size',
        y='cell_number',
        data=df,
        order=list(size_order.keys()),
        hue='base_size',
        palette="Set2",  # 設定顏色樣式
        legend="full",
        width=0.5,        # 控制箱子的寬度
        # flierprops=flierprops
    )
    legend = ax.get_legend()
    handles = legend.legend_handles
    ax.legend(handles, legend_labels, fontsize=20)
    fig.tight_layout()

    # add `data-points` of each size on figure
    for size in size_order.keys():
        # get `x-axis` position
        x_pos = list(size_order.keys()).index(size)
        # get `data-points` for current size
        y_values = df[df['base_size'] == size]['cell_number']
        # scatter plot
        ax.scatter([x_pos] * len(y_values), y_values,
                   color='gray', alpha=0.3, s=20, zorder=5)

    # add `q1`, `q2`, `q3` of each size on figure
    quantile_dicts = {"Q1": q1s, "Q2": q2s, "Q3":q3s}
    for name, quantile_dict in quantile_dicts.items():
        for size in size_order.keys():
            # get `x-axis` position
            x_pos = list(size_order.keys()).index(size)
            # add Q2 (median)
            ax.annotate(f'{name}: {quantile_dict[size]:.2f}',
                        xy=(x_pos, quantile_dict[size]),
                        xytext=(x_pos+0.3, quantile_dict[size]),
                        ha='left',
                        color='black',
                        fontsize=8,
                        zorder=6)

    # plt.show()
    
    # save figure
    result_adv_dir = path_navigator.dbpp.get_one_of_dbpp_roots("result_adv")
    dst_dir = result_adv_dir.joinpath(processed_di.instance_name,
                                      Path(__file__).stem,
                                      seg_desc, seg_dirname)
    create_new_dir(dst_dir)
    cli_out.divide()
    
    for suffix in [".png", ".svg"]:
        save_name = Path(__file__).with_suffix(suffix).name
        fig.savefig(dst_dir.joinpath(save_name))
    print(f"Save 'Avg. cell number under different size' to : '{dst_dir}'")
    
    # save csv file
    save_name = Path(__file__).with_suffix(".csv").name
    df.to_csv(dst_dir.joinpath(save_name), encoding='utf_8_sig', index=False)
    
    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/