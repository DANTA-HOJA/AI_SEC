import os
import sys
from pathlib import Path
import toml

abs_module_path = Path("./../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.data.clustering.surfaceareakmeanscluster import SurfaceAreaKMeansCluster
from modules.shared.config import load_config

# -----------------------------------------------------------------------------------
config_name = "0.6.cluster_data.toml"

config = load_config(config_name)

processed_inst_desc = config["data_processed"]["instance_desc"]

kmeans_rnd = config["param"]["random_seed"]
n_clusters = config["param"]["n_clusters"]
label_str  = config["param"]["label_str"]

old_classdiv_xlsx_path = None
if config["old_classdiv_xlsx"]["full_path"]:
    old_classdiv_xlsx_path = Path(config["old_classdiv_xlsx"]["full_path"])

# -----------------------------------------------------------------------------------
# Initialize a `ProcessedDataInstance` object

processed_data_instance = ProcessedDataInstance()
processed_data_instance.set_attrs(config_name)

if processed_data_instance.data_xlsx_path:
    xlsx_path = processed_data_instance.data_xlsx_path

# -----------------------------------------------------------------------------------
# Generate clustered xlsx
 
for arg_6 in [False, True]:
    for arg_7 in [False, True]: # arg_7: x_axis_log_scale = True，產生的 image 會有 kde curve（比較好區分 "原始刻度" 和 "LOG" 刻度）
        SAKMeansCluster = SurfaceAreaKMeansCluster(xlsx_path, n_clusters, label_str, kmeans_rnd,
                                                   log_base=10, cluster_with_log_scale=arg_6, x_axis_log_scale=arg_7,
                                                   old_classdiv_xlsx_path=old_classdiv_xlsx_path)
        print("="*80, "\n"); print(SAKMeansCluster); SAKMeansCluster.plot_and_save_xlsx()