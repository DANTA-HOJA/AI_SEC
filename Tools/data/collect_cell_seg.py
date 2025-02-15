# %%
import os
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.ml.utils import get_seg_desc
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, exclude_paths, get_repo_root

# -----------------------------------------------------------------------------/
# %%
""" Init components """
path_navigator = PathNavigator()
processed_di = ProcessedDataInstance()
processed_di.parse_config("ml_analysis.toml")

# notebook name
notebook_name = Path(__file__).stem

# -----------------------------------------------------------------------------/
# %%
config = load_config("ml_analysis.toml")
# [seg_results]
seg_desc = get_seg_desc(config)

seg_desc

# -----------------------------------------------------------------------------/
# %%
src_dir = processed_di.palmskin_processed_dir
seg_dirs = exclude_paths(list(src_dir.glob(f"*/{seg_desc}/*")), [".tif"])

seg_dirname_cnt = Counter()
for seg_dir in seg_dirs:
    seg_dirname = seg_dir.name
    seg_dirname_cnt.update([seg_dirname])

seg_dirname_cnt

# -----------------------------------------------------------------------------/
# %%
result_ml_dir = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_ml")
collect_root = result_ml_dir.joinpath(processed_di.instance_name,
                                      f"tool.{notebook_name}",
                                      seg_desc)

for seg_dirname in seg_dirname_cnt.keys():
    
    dst_dir = collect_root.joinpath(seg_dirname)
    create_new_dir(dst_dir)
    
    paths = list(src_dir.glob(f"*/{seg_desc}/{seg_dirname}/*.seg2o.png"))
    for path in paths:
        
        dname = path.relative_to(src_dir).parts[0]
        seg_suffix = path.suffixes[-2]
        
        # get feature
        path_split = deepcopy(list(path.parts))
        path_split[-1] = path_split[-1].replace(f"{seg_suffix}.png", ".ana.toml") # replace file ext.
        toml_file = Path(*path_split) # re-construct path
        cell_cnt = load_config(toml_file)["cell_count"]
        
        # copy and rename image
        new_path = dst_dir.joinpath(f"cnt{cell_cnt}_{dname}.png")
        print(new_path)
        shutil.copy(path, new_path)