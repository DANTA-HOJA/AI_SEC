# Instructions for Deep Learning (DL) Methods

## Create DL Dataset (scripts in `1.make_dataset/`)

```text
1. How to create DL datasets
2. where to save the DL datasets
```

## DL Trianing (scripts in `2.training/`)

```text
1. How to train a DL model
2. where to save the DL models
```

## DL Testing (scripts in `3.test_by_img/` and `4.test_by_fish/`)

```text
1. How to run test (prediction): test_by_img / test_by_fish
2. where to save the DL results
```

## Class Activation Map (Grad-CAM) (scripts in `5.make_cam_gallery/` and `6.run_cam_analysis/`)

```text
1. How to gernerate the CAMs
2. How to make the CAM gallery
3. How to run CAM analysis
```

## Tracking DL Results (script `7.update_db_file/`)

```text
1. How to gernerate a DB-like file for tracking the parameters and model performance (score)
```

## Configuration Settings Examples (.toml)

```text
A series of .ipynb notebooks to demonstrate the valid inputs for configurations.

for example:

script_dl/2.training/2.a.vit_b_16.py <-> script_dl/Config/2.training.toml
script_dl/4.test_by_fish/4.a.vit_b_16.py <-> script_dl/Config/4.test_by_fish.toml
...
```

