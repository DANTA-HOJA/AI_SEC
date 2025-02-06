# AI_SEC

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
- GPU: RTX 4090
- RAM: 64 GB DDR5

## Cellpose Segmentation for Palmskin (2024-12-25)

1. Create an environment follow the setup instructions provided at [Cellpose](https://github.com/MouseLand/cellpose?tab=readme-ov-file#installation)

2. Run the following commands in the "cellpose" environment.

```shell
pip install -U colorama toml tomlkit matplotlib tqdm rich seaborn imagecodecs scikit-image scikit-learn
mamba install imgaug=0.4.0
mamba install numpy=1.23.0
```

## Notification

- At least 150 GB of disk space is required.
- Create the dataset on an SSD to ensure reasonable processing times.
- The entire project uses glob for file searches. If you need to save images in the project folder, please use file formats other than `.tif` or `.tiff`.
