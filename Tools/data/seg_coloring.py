import random
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install
from skimage import color
from skimage.segmentation import mark_boundaries

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.dl.fakepalmskin.utils import gen_unique_random_color_pool
from modules.ml.seg_generate import (average_rgb_coloring, draw_label_on_image,
                                     merge_similar_rgb)

install()
random.seed(2022)
# -----------------------------------------------------------------------------/

uint8_imgs: dict[str, np.ndarray] = {}
float64_imgs: dict[str, np.ndarray] = {}


""" Load `npy` file """
npy_file = Path(r"")
seg1 = np.load(npy_file, allow_pickle=True).item()['masks']

""" Load original image """
# original tiff
img_name = Path(f"{npy_file.stem.replace('_seg', '')}.tiff")
img = ski.io.imread(npy_file.parent.joinpath(img_name))
assert np.iinfo(img.dtype).max == 255 # dtype('uint8')
# update dicts
uint8_imgs["orig"] = deepcopy(img)
float64_imgs["orig"] = deepcopy(img / 255.0)


""" Color with random RGB """
# random coloring
unique_labels = np.unique(seg1)
if 0 in unique_labels:
    unique_labels = unique_labels[1:] # 排除 `background` (0)
color_pool = gen_unique_random_color_pool(unique_labels, lab_lthres=30.0)
colors = np.array(list(color_pool.values()))
img = color.label2rgb(seg1, colors=colors, bg_label=0)
img = draw_label_on_image(seg1, np.uint8(img*255))
# update dicts
uint8_imgs["rand_rgb"] = deepcopy(img)
float64_imgs["rand_rgb"] = deepcopy(img / 255.0)


""" Replace background of segmentation with original pixels """
# replace color
img = deepcopy(uint8_imgs["rand_rgb"])
img[seg1 == 0] = uint8_imgs["orig"][seg1 == 0]
# update dicts
uint8_imgs["rand_rgb.rpbg"] = deepcopy(img)
float64_imgs["rand_rgb.rpbg"] = deepcopy(img / 255.0)


""" Apply `Gamma Correction` to 'rand_rgb.rpbg' """
img = deepcopy(float64_imgs["rand_rgb.rpbg"])
img = np.power(img, 0.5)
# update dicts
float64_imgs["rand_rgb.rpbg.gamma_0d5"] = deepcopy(img)
uint8_imgs["rand_rgb.rpbg.gamma_0d5"] = deepcopy(np.uint8(img*255))


""" Color with average RGB """
img = average_rgb_coloring(seg1, uint8_imgs["orig"])
uint8_imgs["avg_rgb"] = deepcopy(img)
float64_imgs["avg_rgb"] = deepcopy(img / 255.0)


""" Add label on `avg_rgb` """
img = np.uint8(mark_boundaries(float64_imgs["avg_rgb"], seg1, color=(0, 1, 1))*255)
img = draw_label_on_image(seg1, img)
uint8_imgs["seg1a"] = deepcopy(img)
float64_imgs["seg1a"] = deepcopy(img / 255.0)


""" Merge similar RGB """
merge: float = 10.0
debug_mode: bool = True
# merge similar rgb
seg2, relabeling = merge_similar_rgb(seg1, uint8_imgs["orig"],
                                     merge=merge, debug_mode=debug_mode)
# merged segmentation on `orig` and `avg_rgb` images
for k, suffix in {"o": "orig", "a": "avg_rgb"}.items():
    img = np.uint8(mark_boundaries(float64_imgs[suffix], seg2, color=(0, 1, 1))*255)
    img = draw_label_on_image(seg1, img, relabeling=relabeling)
    uint8_imgs[f"seg2{k}"] = deepcopy(img)
    float64_imgs[f"seg2{k}"] = deepcopy(img / 255.0)


""" Save and display images """
for img in float64_imgs.values(): assert img.dtype == np.float64
for img in uint8_imgs.values(): assert img.dtype == np.uint8

# save image
for suffix, img in uint8_imgs.items():
    if suffix == "orig": continue
    ski.io.imsave(npy_file.parent.joinpath(f"{img_name}.{suffix}.png"), img)

# display images
imgset_r1 = ["orig", "rand_rgb", "rand_rgb.rpbg", "rand_rgb.rpbg.gamma_0d5"] # assign image order
imgset_r2 = ["orig", "seg1a", "seg2a", "seg2o"] # assign image order

# create figure
fig, axes = plt.subplots(2, len(imgset_r1))
axes = plt.gcf().get_axes()
for suffix, ax in zip((imgset_r1+imgset_r2), axes):
    ax.imshow(float64_imgs[suffix], vmax=1.0, vmin=0.0)
    ax.set_title(f"{img_name}.{suffix}.png")
plt.tight_layout()
plt.show()