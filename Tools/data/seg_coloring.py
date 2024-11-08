import random
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage as ski
from matplotlib import colors as mcolors
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install
from skimage import color
from skimage.segmentation import mark_boundaries

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.ml.seg_generate import (average_rgb_coloring, draw_label_on_image,
                                     merge_similar_rgb)

install()
np.random.seed(2022)
# -----------------------------------------------------------------------------/

uint8_imgs: dict[str, np.ndarray] = {}
float64_imgs: dict[str, np.ndarray] = {}


""" Load `npy` file """
npy_file = Path(r"")
seg = np.load(npy_file, allow_pickle=True).item()['masks']

""" Load original image """
# original tiff
img_name = f"{npy_file.stem.replace('_seg', '')}.tiff"
img = ski.io.imread(npy_file.parent.joinpath(img_name))
assert np.iinfo(img.dtype).max == 255 # dtype('uint8')
# update dicts
uint8_imgs["orig"] = deepcopy(img)
float64_imgs["orig"] = deepcopy(img / 255.0)


""" Color with random RGB """
# random coloring
colors = np.random.random((len(np.unique(seg)), 3))
img = color.label2rgb(
    seg, colors=colors,
    bg_label=0
)
img = draw_label_on_image(seg, np.uint8(img*255))
# update dicts
uint8_imgs["rand_rgb"] = deepcopy(img)
float64_imgs["rand_rgb"] = deepcopy(img / 255.0)


""" Replace background of segmentation with original pixels """
# replace color
img = deepcopy(float64_imgs["rand_rgb"])
img[seg == 0] = float64_imgs["orig"][seg == 0]
# update dicts
float64_imgs["rand_rgb.rpbg"] = deepcopy(img)
uint8_imgs["rand_rgb.rpbg"] = deepcopy(np.uint8(img*255))


# """ Apply `CLAHE` to 'replaced color' image """
# # convert RGB to LAB
# img = deepcopy(uint8_imgs["rand_rgb.rpbg"])
# lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
# # 分離 L 通道並應用 CLAHE
# l, a, b = cv2.split(lab_img)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# l = clahe.apply(l)
# # 合併調整後的 L 通道與原始的 A 和 B 通道，並轉回 RGB
# lab_img = cv2.merge((l, a, b))
# img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
# # update dicts
# uint8_imgs["clahe"] = deepcopy(img)
# float64_imgs["clahe"] = deepcopy(img / 255.0)


""" Apply `Gamma Correction` to 'rand_rgb.rpbg' """
img = deepcopy(float64_imgs["rand_rgb.rpbg"])
img = np.power(img, 0.5)
# update dicts
float64_imgs["rand_rgb.rpbg.gamma_0d5"] = deepcopy(img)
uint8_imgs["rand_rgb.rpbg.gamma_0d5"] = deepcopy(np.uint8(img*255))


""" Color with average RGB """
img = average_rgb_coloring(seg, uint8_imgs["orig"])
uint8_imgs["avg_rgb"] = deepcopy(img)
float64_imgs["avg_rgb"] = deepcopy(img / 255.0)


""" Merge similar RGB """
merge_thres: float = 10.0
debug_mode: bool = True
# merge similar rgb
merged_seg, relabeling = merge_similar_rgb(seg, uint8_imgs["orig"],
                                           merge_thres, debug_mode=debug_mode)
# merged segmentation on `orig` and `avg_rgb` images
for suffix in ["orig", "avg_rgb"]:
    img = deepcopy(uint8_imgs[suffix])
    for label in relabeling.values():
        mask = (merged_seg == label)
        img = np.uint8(mark_boundaries(img, mask)*255)
    img = draw_label_on_image(seg, img, relabeling=relabeling)
    uint8_imgs[f"merged_seg_on_{suffix}"] = deepcopy(img)
    float64_imgs[f"merged_seg_on_{suffix}"] = deepcopy(img / 255.0)


""" Add label on `avg_rgb` """
img = draw_label_on_image(seg, uint8_imgs["avg_rgb"])
uint8_imgs["avg_rgb"] = deepcopy(img)
float64_imgs["avg_rgb"] = deepcopy(img / 255.0)


""" Save and display images """
for img in float64_imgs.values(): assert img.dtype == np.float64
for img in uint8_imgs.values(): assert img.dtype == np.uint8

# save image
for suffix, img in uint8_imgs.items():
    if suffix == "orig": continue
    ski.io.imsave(npy_file.with_suffix(f".{suffix}.png"), img)

# display images
imgset_r1 = ["orig", "rand_rgb", "rand_rgb.rpbg", "rand_rgb.rpbg.gamma_0d5"] # assign image order
imgset_r2 = ["orig", "avg_rgb", "merged_seg_on_avg_rgb", "merged_seg_on_orig"] # assign image order

# create figure
fig, axes = plt.subplots(2, len(imgset_r1))
axes = plt.gcf().get_axes()
for suffix, ax in zip((imgset_r1+imgset_r2), axes):
    ax.imshow(float64_imgs[suffix], vmax=1.0, vmin=0.0)
    ax.set_title(npy_file.with_suffix(f".{suffix}.png").name)
plt.tight_layout()
plt.show()