# Instructions for Data Preprocessing

## Data Structure

```text
{`note`}_Academia_Sinica_i[Num]/
    |
    |--- {`bf_param_note`}_BrightField_analyze/
    |--- {`bf_param_note`}_BrightField_reCollection/
    |--- {`palmskin_param_note`}_PalmSkin_preprocess/
    |--- {`palmskin_param_note`}_PalmSkin_reCollection/
    |--- Clustered_File/
    |--- data.csv
    |--- datasplit_[`RND`].csv
    |--- split_count.log
```

## Image Processing

### Palmskin Image (Per Image)

- **Included in the deposited data**

#### Converting 3D `LIF` to 2D `TIFF` (script `0.2.1`)

- Script: [0.2.1.preprocess_palmskin.py](script_data/0.2.1.preprocess_palmskin.py)
- Config: [Config/0.2.1.preprocess_palmskin.toml](script_data/Config/0.2.1.preprocess_palmskin.toml)
```text
1. How to create
2. where to save
```

### Bright-field Image (Per Fish)

- **Included in the deposited data**

#### 1. Cropping and Converting from 16-bit to 8-bit Image (script `0.3.1`)

```text
1. How to create
2. where to save
```

#### 2. Res18-Unet Automated Segmentation (refer to [`script_bfseg/`]())

```text
1. How to create
2. where to save
```

#### 3. Obtaining the standard length and trunk surface area (script `0.3.2`)

```text
1. How to create
2. where to save
```

## File Collector (script `0.4.1`)

```text
Rearrange the results together (for easier viewing)
```

## Clustered File (scripts `0.5.x`)

### 1. create `data.csv`

```text
columns:

Brightfield
Analysis Mode
Palmskin Anterior (SP8)
Palmskin Posterior (SP8)
"Trunk surface area, SA (um2)"
"Standard Length, SL (um)"
```

### 2. split train/valid/test

```text
1. How to create
2. where to save
```

### 3. generate size label (clustering)

```text
1. How to create
2. where to save
```

## Rescaled Bright-field to a Uniform Size (script `0.6.1`)

```text
1. How to create
2. where to save
```

## Configuration Settings Examples (*.toml)

```text
A series of .ipynb notebooks to demonstrate the valid inputs for configurations.
...
```
