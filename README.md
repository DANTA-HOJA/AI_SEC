# AI_SEC

```text
1. Need one sentence to briefly describe this research.
2. Manual script DOI.
3. Link to deposited data.
```

## Package Installation Test (2024-06-03)

```shell
mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install scikit-image==0.22.0 scikit-learn==1.4.0
pip install -U colorama toml tomlkit matplotlib tqdm rich seaborn imagecodecs
mamba install pandas pyimagej openjdk=8 imgaug=0.4.0
pip install albumentations==1.3.1
pip install grad-cam==1.4.8
mamba install numpy=1.23.0
mamba install mkl==2024.0
pip install umap-learn==0.5.6
```

## Hardware Information

- OS: Ubuntu 22.04
- CPU: i9-13900KS
- GPU: RTX-4090 with 24GB VRAM
- RAM: 64 GB DDR5

## Notifications

- Do NOT modify the name of this repository. If you really need to rename the repository, remember to update the list of strings in [`utils.py`](modules/shared/utils.py#L58).
- At least 150 GB of disk space is required.
- Create the dataset on an SSD or using an OS like Linux with aggressive caching to ensure reasonable processing times.
- The entire project uses [`glob`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob) for file searches. If you need to save images in the project folder, please use file formats other than `.tif` or `.tiff`.

## File Structure

`Tools/` : The scripts are for debugging. **Please do not run them unless necessary; use at your own risk.**

`dev/` : The files are under development or already deprecated, so they may not work. **Please do not run them unless necessary; use at your own risk.**

`modules/` : All the utility functions and classes.

`script_data/` : Converts `.lif` files into commonly used image formats, splits into train/validation/test sets, applies K-Means, etc.

`script_bfseg/` : ResNet18-UNet automated segmentation for obtaining the "trunk surface area". Inspired by [`usuyama/pytorch-unet`](https://github.com/usuyama/pytorch-unet). **(merging into this repo later)**

`script_ml/` : From Cellpose segmentation to simple features, machine learning results, and UMAP. Inspired by [`ccsyan/labeling-cells-using-slic`](https://github.com/ccsyan/labeling-cells-using-slic).

`script_dl/` : Everything related to deep learning, including CAM analysis.

`script_adv/` : Advanced results, such as "critical SEC number" and "feature-subtracted images", etc.

** For detailed instructions for each part, please click on the corresponding directory (folder).

## `db_path_plan.toml`

A file that handles the linking of data and code, allowing them to be stored separately, i.e., enabling the data to be placed anywhere on your system.

```text
descriptions for the file (每個 folder 對應甚麼要講清楚)
```

** For detailed instructions for each part, please click on the corresponding directory (folder).

## (TBD) Need to choose a licence?

之前 `ccsyan/labeling-cells-using-slic` 有放，但我對這塊不太熟

## (TBD) Need to put References?

之前 `ccsyan/labeling-cells-using-slic` 有放，還是其實不用因為 paper 有了?

## (TBD) Feedback?

之前 `ccsyan/labeling-cells-using-slic` 有以下文字：

**Made changes to the layout templates or some other part of the code? Fork this repository, make your changes, and send a pull request. Do these codes help on your research? Please cite as the follows. Skin cells undergo XXXXXXX. KY Chan, CCS Yan, HY Roan, SC Hsu, CD Hsiao, CP Hsu, CH Chen.**
