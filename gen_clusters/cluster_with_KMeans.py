import sys
from pathlib import Path
import toml

rel_module_path = "./../modules/"
sys.path.append( str(Path(rel_module_path).resolve()) ) # add path to scan customized module

from clustering.SurfaceAreaKMeansCluster import SurfaceAreaKMeansCluster


# -----------------------------------------------------------------------------------
# Load `db_path_plan.toml`
with open("./../Config/db_path_plan.toml", mode="r") as f_reader:
    dbpp_config = toml.load(f_reader)
db_root = Path(dbpp_config["root"])


# -----------------------------------------------------------------------------------
# Load `(Cluster)_data.toml`
with open("./../Config/(Cluster)_data.toml", mode="r") as f_reader:
    config = toml.load(f_reader)

preprocessed_desc = config["data_preprocessed"]["desc"]

if not config["old_classdiv_xlsx"]["full_path"]: old_classdiv_xlsx_path=None
else: old_classdiv_xlsx_path = Path(config["old_classdiv_xlsx"]["full_path"])

kmeans_rnd = config["param"]["random_seed"]
n_clusters = config["param"]["n_clusters"]
label_str  = config["param"]["label_str"]


# -----------------------------------------------------------------------------------
# Generate `path_vars`

# Check `{desc}_Academia_Sinica_i[num]`
data_root = db_root.joinpath(dbpp_config["data_preprocessed"])
target_dir_list = list(data_root.glob(f"*{preprocessed_desc}*"))
assert len(target_dir_list) == 1, (f"[data_preprocessed.desc] in `(Cluster)_data.toml` is not unique/exists, "
                                   f"find {len(target_dir_list)} possible directories, {target_dir_list}")
preprocessed_root = target_dir_list[0]

# xlsx: .../{Data}_Preprocessed/{desc}_Academia_Sinica_i[num]/data.xlsx
xlsx_path = preprocessed_root.joinpath("data.xlsx")


# -----------------------------------------------------------------------------------
for arg_6 in [False, True]:
    for arg_7 in [False, True]: # arg_7: x_axis_log_scale = True，產生的 image 會有 kde curve（比較好區分 "原始刻度" 和 "LOG" 刻度）
        SAKMeansCluster = SurfaceAreaKMeansCluster(xlsx_path, n_clusters, label_str, kmeans_rnd,
                                                   log_base=10, cluster_with_log_scale=arg_6, x_axis_log_scale=arg_7,
                                                   old_classdiv_xlsx_path=old_classdiv_xlsx_path)
        print("="*80, "\n"); print(SAKMeansCluster); SAKMeansCluster.plot_and_save_xlsx()