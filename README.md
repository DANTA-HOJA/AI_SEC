# ZebraFish_DL

MD705 cooperation project ( zebrafish size classifier by ```Anterior```, ```Posterior``` )

## Package Dependencies

```text
numpy = 1.23.0
pytorch = 1.13.1 + cu116
scikit-image = 0.22.0
scikit-learn = 1.4.0
imgaug = 0.4.0
albumentations = 1.3.1
grad-cam = 1.4.8
```

## Package Installation Test (20240603)

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

## Hardware (20240603)

- OS: Ubuntu 22.04
- CPU: i9-13900KS
- GPU: RTX 4090
- RAM: 64 GB DDR5

## Cellpose Segmentation for Palmskin (20241225)

1. Create an environment follow the setup instructions provided at [Cellpose](https://github.com/MouseLand/cellpose?tab=readme-ov-file#installation)

2. Run the following commands in the "cellpose" environment.

```shell
pip install -U colorama toml tomlkit matplotlib tqdm rich seaborn imagecodecs scikit-image scikit-learn
mamba install imgaug=0.4.0
mamba install numpy=1.23.0
```

## Notification (20240603)

- palmskin_raw_lif : ```[ Ch1, Ch2, Ch3, Ch4 ] -> [ B, G, R, (BF) ]```
- create dataset on SSD to get a reasonable process time
- 全專案使用 glob 搜尋檔案，若要再專案資料夾下儲存影像，請使用  .tif / .tiff 以外的檔案格式。
