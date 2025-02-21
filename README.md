# AI_SEC

This study utilizes different machine learning and deep learning models to provide a proof of concept that organismic information at a macroscopic length scale may be de-encrypted from a snapshot of only a few dozen cells.

- `Manual script DOI.`
- [![DOI](https://img.shields.io/badge/DOI-10.6019%2FS--BIAD1654-blue)][data_doi]

[data_doi]: https://doi.org/10.6019/S-BIAD1654

## Notifications

- Do ***NOT*** modify the name of this repository. If you really need to rename the repository, remember to update the list of strings in [`utils.py`](modules/shared/utils.py#L58).
- At least ***150 GB*** of disk space is required.
- Do ***NOT*** run a script with a larger number before ensuring that all scripts with smaller numbers have been executed, i.e., always execute scripts in **sequential order**.
- Create the dataset on an **SSD** or using an OS like Linux with **aggressive caching** to ensure reasonable processing times.
- The entire project uses [`glob`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob) for file searches. If you need to save images in the project folder, please use file formats other than `.tif` or `.tiff`.

## System Information

- CPU: i9-13900KS
- GPU: RTX-4090 with 24GB VRAM
- RAM: 64 GB DDR5
- OS: Ubuntu 22.04
- Python Version: 3.10

## Data

The data is available from [`DOI:10.6019/S-BIAD1654`][data_doi]

## Installation

1. Follow the instructions [here][miniforge-repo] to install `Miniforge`.
2. Create a new environment and activate it:

    ```shell
    mamba create -n AI_SEC python=3.10
    mamba activate AI_SEC
    ```

3. Install the required packages (Tested on 2024-06-03):

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

4. Download the [`Fiji (ImageJ)`][fiji-dllink] and udpate the ***absolute path*** to `db_path_plan.toml` (see the `db_path_plan.toml` section below).

5. Install the `Fiji (ImageJ)` plugin [`Find Focused Slices`][find-focused-web]

   1. Download the plugin file [`Find_focused_slices.class`][find-focused-dllink]
   2. Place it into the `plugins` folder of your `Fiji (ImageJ)` directory:

        ```text
        📂 Fiji.app/
        ├── 📂 ...
        ├── 📂 plugins/
        │   ├── 📂 ...
        │   ├── 📄 Find_focused_slices.class (◀️ Place the file here)
        │   └── 📄 ...
        └── ...
        ```

6. Download and unzip [`our data`][data_doi], then organize it into the following structure:

    ```text
    📂 AISEC.data/
    ├── 📂 {Data}_Processed/
    ├── 📂 {Dataset}_DL/
    ├── 📂 {Dataset}_ML/
    ├── 📂 {Model}_BFSeg/
    ├── 📂 {Model}_Cellpose/
    ├── 📂 {Model}_DL/
    ├── 📂 {Results}_Advanced/ (◀️ manually create the empty folder)
    ├── 📂 {Results}_DL/ (◀️ manually create the empty folder)
    └── 📂 {Results}_ML/ (◀️ manually create the empty folder)
    ```

    > Note: To avoid encountering a `FileNotFoundError`, manually create the `{Results}_Advanced/`, `{Results}_DL/`, and `{Results}_ML/` directories after unzipping.

7. Update the ***absolute path*** in `db_path_plan.toml` (see the `db_path_plan.toml` section below).

[miniforge-repo]: https://github.com/conda-forge/miniforge
[fiji-dllink]: https://imagej.net/software/fiji/
[find-focused-web]: https://sites.google.com/site/qingzongtseng/find-focus
[find-focused-dllink]: https://github.com/qztseng/imagej_plugins/raw/master/current/Find_focused_slices.class

## File Structure

`Tools/` : The scripts are for debugging. ***Please do not run them unless necessary; use at your own risk.***

`dev/` : The files are under development or already deprecated, so they may not work. ***Please do not run them unless necessary; use at your own risk.***

`modules/` : All the utility functions and classes.

`script_data/` : Converts `.lif` files into commonly used image formats, splits into train/validation/test sets, applies K-Means, etc.

`script_bfseg/` : ResNet18-UNet automated segmentation for obtaining the "trunk surface area". Inspired by [`usuyama/pytorch-unet`](https://github.com/usuyama/pytorch-unet).

`script_ml/` : From Cellpose segmentation to simple features, machine learning results, and UMAP. Inspired by [`ccsyan/labeling-cells-using-slic`](https://github.com/ccsyan/labeling-cells-using-slic).

`script_dl/` : Everything related to deep learning, including CAM analysis.

`script_adv/` : Advanced results, such as "critical SEC number" and "feature-subtracted images", etc.

## Usage

** For detailed instructions on each part, please click on the corresponding directory (folder) listed in the section above.

## `db_path_plan.toml`

The `TOML` file manages the linkage between data and code, allowing them to be stored separately. This setup enables you to place the [`our data`][data_doi] anywhere on your system.

It is parsed as a directory within the program:

- The left side (key) represents the variable used in the code.
- The right side (value) represents the name of the corresponding `data` directory.

```text
root = "" # absolute path of the data directory on this computer

data_processed      = "{Data}_Processed"
dataset_cropped_v3  = "{Dataset}_DL"
dataset_ml          = "{Dataset}_ML"
model_bfseg         = "{Model}_BFSeg"
model_cellpose      = "{Model}_Cellpose"
model_history       = "{Model}_DL"
model_prediction    = "{Results}_DL"
result_ml           = "{Results}_ML"
result_adv          = "{Results}_Advanced"

fiji_local = "" # absolute path of "Fiji (ImageJ)" on this computer
```

- The `root` key should be assigned to the ***absolute path*** of the data directory on your system.
- The `fiji_local` key should be assigned to the ***absolute path*** of `Fiji (ImageJ)` on your system.
- The names of the `data` directories are adjustable. If you change them, remember to update the corresponding names in the `TOML` file.

> Note: Key-value pairs that appear in the `TOML` file but are not mentioned here do not affect your experience with this project.

## Thanks

- <https://sites.google.com/site/qingzongtseng/find-focus>
- <https://github.com/ccsyan/labeling-cells-using-slic>
- <https://github.com/usuyama/pytorch-unet>
