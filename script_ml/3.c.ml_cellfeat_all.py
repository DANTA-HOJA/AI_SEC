# %%
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rich
from rich.pretty import Pretty
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.tester.utils import confusion_matrix_with_class
from modules.ml.utils import (get_cellpose_param_name, get_seg_desc,
                              get_slic_param_name,
                              save_confusion_matrix_display)
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir

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
# load config
config = load_config("ml_analysis.toml")
# [data_processed]
palmskin_result_name: Path = Path(config["data_processed"]["palmskin_result_name"])
cluster_desc: str = config["data_processed"]["cluster_desc"]
# [seg_results]
seg_desc = get_seg_desc(config)
# [Cellpose]
cp_model_name: str = config["Cellpose"]["cp_model_name"]
rich.print("", Pretty(config, expand_all=True))

# -----------------------------------------------------------------------------/
# %%
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

# -----------------------------------------------------------------------------/
# %%
# csv file
dataset_ml_dir = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_ml")
ml_csv = dataset_ml_dir.joinpath(processed_di.instance_name,
                                 cluster_desc,
                                 seg_desc, seg_dirname,
                                 "ml_dataset.csv")

# dst dir
result_ml_dir = path_navigator.dbpp.get_one_of_dbpp_roots("result_ml")
dst_dir = result_ml_dir.joinpath(processed_di.instance_name,
                                 cluster_desc,
                                 seg_desc, seg_dirname)

# -----------------------------------------------------------------------------/
# %%
df = pd.read_csv(ml_csv, encoding='utf_8_sig')
print(f"Read ML Dataset: '{ml_csv}'")
df

# -----------------------------------------------------------------------------/
# %%
labels = sorted(Counter(df["class"]).keys())
label2idx = {label: idx for idx, label in enumerate(labels)}
rich.print(f"labels = {labels}")
rich.print(f"label2idx = {label2idx}")

# -----------------------------------------------------------------------------/
# %%
training_df = df[(df["dataset"] == "train") | (df["dataset"] == "valid")]
test_df = df[(df["dataset"] == "test")]

training_df

# -----------------------------------------------------------------------------/
# %% [markdown]
# ## Training

# -----------------------------------------------------------------------------/
# %%
input_training = training_df.iloc[:, 3:].to_numpy()
dst_dir = dst_dir.joinpath(notebook_name)
create_new_dir(dst_dir)

# 初始化 Random Forest 分類器
rand_seed = int(cluster_desc.split("_")[-1].replace("RND", ""))
random_forest = RandomForestClassifier(n_estimators=100, random_state=rand_seed)

# 訓練模型
idx_gt_training = [label2idx[c_label] for c_label in training_df["class"]]
random_forest.fit(input_training, idx_gt_training)

# get auto tree depths
tree_depths = {}
for i, tree in enumerate(random_forest.estimators_):
    tree_depths[f"Tree {i+1} depth"] = tree.tree_.max_depth

tree_depths["mean depth"] = np.mean(list(tree_depths.values()))
rich.print(f"-> mean of tree depths: {tree_depths['mean depth']}")

tree_depths["median depth"] = np.median(list(tree_depths.values()))
rich.print(f"-> median of tree depths: {tree_depths['median depth']}")

with open(dst_dir.joinpath(f"{notebook_name}.tree_depths.log"), mode="w") as f_writer:
    json.dump(tree_depths, f_writer, indent=4)

# -----------------------------------------------------------------------------/
# %%
# 預測訓練集
pred_train = random_forest.predict(input_training)
pred_train = [labels[c_idx] for c_idx in pred_train]

gt_training = list(training_df["class"])

# reports
cls_report = classification_report(y_true=gt_training,
                                   y_pred=pred_train, digits=5)
_, confusion_matrix = confusion_matrix_with_class(prediction=pred_train,
                                                  ground_truth=gt_training)
# display report
print("Classification Report:\n\n", cls_report)
rich.print(f"{confusion_matrix}\n")

# log file
with open(dst_dir.joinpath(f"{notebook_name}.train.log"), mode="w") as f_writer:
    f_writer.write("Classification Report:\n\n")
    f_writer.write(f"{cls_report}\n\n")
    f_writer.write(f"{confusion_matrix}\n")

# -----------------------------------------------------------------------------/
# %% [markdown]
# ## Test

# -----------------------------------------------------------------------------/
# %%
input_test = test_df.iloc[:, 3:].to_numpy()

# 預測測試集
pred_test = random_forest.predict(input_test)
pred_test = [labels[c_idx] for c_idx in pred_test]

gt_test = list(test_df["class"])

# reports
cls_report = classification_report(y_true=gt_test,
                                   y_pred=pred_test, digits=5)
_, confusion_matrix = confusion_matrix_with_class(prediction=pred_test,
                                                  ground_truth=gt_test)
# display report
print("Classification Report:\n\n", cls_report)
rich.print(f"{confusion_matrix}\n")

# log file
with open(dst_dir.joinpath(f"{notebook_name}.test.log"), mode="w") as f_writer:
    f_writer.write("Classification Report:\n\n")
    f_writer.write(f"{cls_report}\n\n")
    f_writer.write(f"{confusion_matrix}\n")

# Confusion Matrix (image ver.)
save_confusion_matrix_display(y_true=gt_test,
                              y_pred=pred_test,
                              save_path=dst_dir,
                              feature_desc=notebook_name,
                              dataset_desc="test")

print(f"Results Save Dir: '{dst_dir}'")

# -----------------------------------------------------------------------------/
# %%
np.array(pred_test)

# -----------------------------------------------------------------------------/
# %%
np.array(gt_test)