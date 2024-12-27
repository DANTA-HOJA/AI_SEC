import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from rich import print
from rich.pretty import Pretty
from rich.traceback import install

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data import dname
from modules.data.processeddatainstance import ProcessedDataInstance
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
    cluster_desc: str = config["data_processed"]["cluster_desc"]
    # [seg_results]
    seg_desc = get_seg_desc(config)
    # [Cellpose]
    cp_model_name: str = config["Cellpose"]["cp_model_name"]
    # [ML]
    max_topn_patch = config["ML"]["max_topn_patch"]
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
    
    # load `clustered file`
    csv_path = processed_di.clustered_files_dict[cluster_desc]
    clustered_df: pd.DataFrame = pd.read_csv(csv_path, encoding='utf_8_sig', index_col=[0])
    clustered_df["fish_id"] = clustered_df["Brightfield"].apply(lambda x: dname.get_dname_sortinfo(x)[0])
    clustered_df = clustered_df.set_index("fish_id")
    palmskin_dnames = sorted(pd.concat([clustered_df["Palmskin Anterior (SP8)"],
                                        clustered_df["Palmskin Posterior (SP8)"]]), key=dname.get_dname_sortinfo)
    
    # collect informations
    dataset_df = pd.DataFrame()
    for palmskin_dname in palmskin_dnames:
        # prepare
        path = processed_di.palmskin_processed_dir.joinpath(palmskin_dname)
        seg_analysis = path.joinpath(seg_desc, seg_dirname,
                                     f"{seg_dirname}.ana.toml")
        seg_analysis = load_config(seg_analysis)
        fish_id = dname.get_dname_sortinfo(palmskin_dname)[0]
        
        # >>> Create `temp_dict` <<<
        temp_dict = {}
        # -------------------------------------------------------
        temp_dict["palmskin_dname"] = palmskin_dname
        temp_dict["class"] = clustered_df.loc[fish_id, "class"]
        temp_dict["dataset"] = clustered_df.loc[fish_id, "dataset"]
        # -------------------------------------------------------
        for k, v in seg_analysis.items():
            if k == f"patch_sizes": continue
            temp_dict[k] = v
        
        for i, patch_size in enumerate(seg_analysis[f"patch_sizes"], start=1):
            temp_dict[f"top{i}_patch"] = patch_size
            if i == max_topn_patch: break
        # -------------------------------------------------------
        
        temp_df = pd.DataFrame(temp_dict, index=[0])
        if dataset_df.empty: dataset_df = temp_df.copy()
        else: dataset_df = pd.concat([dataset_df, temp_df], ignore_index=True)
    
    # drop columns if any NAN values
    dataset_df = dataset_df.dropna(axis=1)
    
    # save Dataframe as a CSV file (for segmentation)
    result_ml_dir = path_navigator.dbpp.get_one_of_dbpp_roots("result_ml")
    save_path = result_ml_dir.joinpath("Generated",
                                       processed_di.instance_name, cluster_desc,
                                       seg_desc, seg_dirname,
                                       "ml_dataset.csv")
    create_new_dir(save_path.parent)
    dataset_df.to_csv(save_path, encoding='utf_8_sig', index=False)
    print(f"ML_table (for segmentation): '{save_path}'\n")
    
    # save Dataframe as a CSV file (for image PCA reduction)
    dataset_df = dataset_df.iloc[:, :3]
    result_ml_dir = path_navigator.dbpp.get_one_of_dbpp_roots("result_ml")
    save_path = result_ml_dir.joinpath("Generated",
                                       processed_di.instance_name, cluster_desc,
                                       "ImagePCA", "ml_dataset.csv")
    create_new_dir(save_path.parent)
    dataset_df.to_csv(save_path, encoding='utf_8_sig', index=False)
    print(f"ML_table (for image PCA reduction): '{save_path}'\n")
    # -------------------------------------------------------------------------/