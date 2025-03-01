# %%
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rich import print
from rich.pretty import Pretty
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.tester.utils import confusion_matrix_with_class
from modules.ml.utils import save_confusion_matrix_display
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
print("", Pretty(config, expand_all=True))

# -----------------------------------------------------------------------------/
# %%
# csv file
dataset_ml_dir = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_ml")
ml_csv = dataset_ml_dir.joinpath(processed_di.instance_name,
                                 cluster_desc,
                                 "ImagePCA",
                                 "ml_dataset.csv")

# dst
result_ml_dir = path_navigator.dbpp.get_one_of_dbpp_roots("result_ml")
dst_dir = result_ml_dir.joinpath(processed_di.instance_name,
                                 cluster_desc,
                                 "ImagePCA")

# -----------------------------------------------------------------------------/
# %%
df = pd.read_csv(ml_csv, encoding='utf_8_sig')
print(f"Read ML Dataset: '{ml_csv}'")
df

# -----------------------------------------------------------------------------/
# %%
labels = sorted(Counter(df["class"]).keys())
label2idx = {label: idx for idx, label in enumerate(labels)}
print(f"labels = {labels}")
print(f"label2idx = {label2idx}")

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
n_pca = 5
img_mode: str = config["ML"]["img_mode"]
dst_dir = dst_dir.joinpath(f"{palmskin_result_name.stem}.First{n_pca}PCA/{notebook_name}.{img_mode}")
create_new_dir(dst_dir)

palmskin_result_name = Path(f"{palmskin_result_name.stem}.W512_H1024.tif")
rel_path, _ = processed_di.get_sorted_results_dict("palmskin", str(palmskin_result_name))
print(f"[yellow]{rel_path}")

img_resize: tuple = tuple(config["ML"]["img_resize"])
img_fullsize = None
data = []
for palmskin_dname in tqdm(training_df["palmskin_dname"]):
    img_path: Path = processed_di.palmskin_processed_dir.joinpath(palmskin_dname, rel_path)
    image = cv2.imread(str(img_path))
    if img_fullsize is None:
        img_fullsize = image.shape[:2][::-1]
    if image is not None:
        image = cv2.cvtColor(image, getattr(cv2, f"COLOR_BGR2{img_mode}"))
        image = cv2.resize(image, img_resize, interpolation=cv2.INTER_CUBIC)
        image = image.flatten()
        data.append([image, img_path])

features, img_paths  = zip(*data)

# -----------------------------------------------------------------------------/
# %%
rand_seed = int(cluster_desc.split("_")[-1].replace("RND", ""))

# image pca
features = np.array(features) # convert type, shape = (n, 45*45*3)
pca = PCA(n_components=n_pca, random_state=rand_seed)
pca.fit(features)
input_training = pca.transform(features)

# 初始化 Random Forest 分類器
random_forest = RandomForestClassifier(n_estimators=100, random_state=rand_seed)

# 訓練模型
idx_gt_training = [label2idx[c_label] for c_label in training_df["class"]]
random_forest.fit(input_training, idx_gt_training)

# get auto tree depths
print("\n", end="")

tree_depths = {}
for i, tree in enumerate(random_forest.estimators_):
    tree_depths[f"Tree {i+1} depth"] = tree.tree_.max_depth

tree_depths["mean depth"] = np.mean(list(tree_depths.values()))
print(f"-> mean of tree depths: {tree_depths['mean depth']}")

tree_depths["median depth"] = np.median(list(tree_depths.values()))
print(f"-> median of tree depths: {tree_depths['median depth']}")

with open(dst_dir.joinpath(f"{notebook_name}.{img_mode}.tree_depths.log"), mode="w") as f_writer:
    json.dump(tree_depths, f_writer, indent=4)

# -----------------------------------------------------------------------------/
# %%
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

eigenvalues = pca.explained_variance_
prop_vars = pca.explained_variance_ratio_
pca_n_comps = pca.n_components_

plt.plot(np.arange(pca_n_comps), prop_vars, 'ro-')
plt.xlabel("PCA Components")
plt.xticks(range(pca_n_comps), range(1, pca_n_comps+1))
plt.title("Explained Variance Ratio")
plt.savefig(dst_dir.joinpath(f"{pca_n_comps}c_explained_variance.png"))

accum_prop_var = 0
for i, prop_var in enumerate(prop_vars, start=1):
    accum_prop_var += prop_var
    if accum_prop_var > 0.9:
        print(f"{i} PCA components, accum_prop_var: {accum_prop_var}")
        break

print("\n", end="")
print(f"-> {pca_n_comps} PCA components")
print(f"-> prop_vars: {prop_vars}")
print(f"-> accum_prop_var: {np.sum(prop_vars)}")
print(f"-> pca feature importances to model: {random_forest.feature_importances_}")
print("\n", end="")

# -----------------------------------------------------------------------------/
# %%
fig, axes = plt.subplots(1, pca_n_comps, figsize=(pca_n_comps, 2), dpi=1000)
assert len(axes.flatten()) == len(pca.components_)

img_reducesize = np.uint(img_fullsize/np.gcd.reduce(img_fullsize)*200)
for i, ax in enumerate(axes.flatten()):
    pc = pca.components_[i].reshape(*img_resize[::-1], 3)
    pc = cv2.resize(pc, img_reducesize, interpolation=cv2.INTER_CUBIC)
    pc = (pc-np.min(pc)) / (np.max(pc) - np.min(pc))
    ax.imshow(pc)
    ax.set_title("pc{} ({:.4f})".format(i+1, random_forest.feature_importances_[i]), fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(dst_dir.joinpath(f"{pca_n_comps}c_feature_importances.png"))

# -----------------------------------------------------------------------------/
# %%
fig2, axes2 = plt.subplots(1, 2, figsize=(2, 2), dpi=1000)

# normal
pc = np.mean((pca.components_), axis=0)
pc = pc.reshape(*img_resize[::-1], 3)
pc = cv2.resize(pc, img_reducesize, interpolation=cv2.INTER_CUBIC)
pc = (pc-np.min(pc)) / (np.max(pc) - np.min(pc))
axes2[0].imshow(pc)
axes2[0].set_title("normal")
axes2[0].set_xticks([])
axes2[0].set_yticks([])

# weighted
from copy import deepcopy

pca_weighted_c = deepcopy(pca.components_)
for i, (pc, fimp) in enumerate(zip(pca_weighted_c, random_forest.feature_importances_)):
    pca_weighted_c[i] = pc*fimp

pc = np.mean(pca_weighted_c, axis=0)
pc = pc.reshape(*img_resize[::-1], 3)
pc = cv2.resize(pc, img_reducesize, interpolation=cv2.INTER_CUBIC)
pc = (pc-np.min(pc)) / (np.max(pc) - np.min(pc))
axes2[1].imshow(pc)
axes2[1].set_title("weighted")
axes2[1].set_xticks([])
axes2[1].set_yticks([])

fig2.savefig(dst_dir.joinpath(f"{pca_n_comps}c_feature_importances_comp.png"))

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
print(f"{confusion_matrix}\n")

# log file
with open(dst_dir.joinpath(f"{notebook_name}.{img_mode}.train.log"), mode="w") as f_writer:
    f_writer.write("Classification Report:\n\n")
    f_writer.write(f"{cls_report}\n\n")
    f_writer.write(f"{confusion_matrix}\n")

# -----------------------------------------------------------------------------/
# %% [markdown]
# ## Test

# -----------------------------------------------------------------------------/
# %%
data = []
for palmskin_dname in tqdm(test_df["palmskin_dname"]):
    img_path: Path = processed_di.palmskin_processed_dir.joinpath(palmskin_dname, rel_path)
    image = cv2.imread(str(img_path))
    if image is not None:
        image = cv2.cvtColor(image, getattr(cv2, f"COLOR_BGR2{img_mode}"))
        image = cv2.resize(image, img_resize, interpolation=cv2.INTER_CUBIC)
        image = image.flatten()
        data.append([image, img_path])

features, img_paths  = zip(*data)

# -----------------------------------------------------------------------------/
# %%
input_test = pca.transform(features)

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
print(f"{confusion_matrix}\n")

# log file
with open(dst_dir.joinpath(f"{notebook_name}.{img_mode}.test.log"), mode="w") as f_writer:
    f_writer.write("Classification Report:\n\n")
    f_writer.write(f"{cls_report}\n\n")
    f_writer.write(f"{confusion_matrix}\n")

# Confusion Matrix (image ver.)
save_confusion_matrix_display(y_true=gt_test,
                              y_pred=pred_test,
                              save_path=dst_dir,
                              feature_desc=f"{notebook_name}.{img_mode}",
                              dataset_desc="test")

print(f"Results Save Dir: '{dst_dir}'")

# -----------------------------------------------------------------------------/
# %%
np.array(pred_test)

# -----------------------------------------------------------------------------/
# %%
np.array(gt_test)