# Result Names (20231229 UPDATE)

## PalmSkin_preprocess

```text
03_RGB_direct_max_zproj.tif (Do Nothing)

--------------------------------------------------------------------------------
# B channel (9 files)

MetaImage\00_ch_B_direct.tif
MetaImage\04_ch_B_m3d.tif
MetaImage\05_ch_B_mm3d.tif
MetaImage\06_ch_B_mm3d_kuwahara.tif
MetaImage\08_ch_B_m3d_HE.tif
MetaImage\09_ch_B_mm3d_HE.tif
MetaImage\10_ch_B_mm3d_kuwahara_HE.tif
07_ch_B_fusion.tif
11_ch_B_HE_fusion.tif

--------------------------------------------------------------------------------
# G channel (9 files)

MetaImage\01_ch_G_direct.tif
MetaImage\12_ch_G_m3d.tif
MetaImage\13_ch_G_mm3d.tif
MetaImage\14_ch_G_mm3d_kuwahara.tif
MetaImage\16_ch_G_m3d_HE.tif
MetaImage\17_ch_G_mm3d_HE.tif
MetaImage\18_ch_G_mm3d_kuwahara_HE.tif
15_ch_G_fusion.tif
19_ch_G_HE_fusion.tif

--------------------------------------------------------------------------------
# R channel (9 files)

MetaImage\02_ch_R_direct.tif
MetaImage\20_ch_R_m3d.tif
MetaImage\21_ch_R_mm3d.tif
MetaImage\22_ch_R_mm3d_kuwahara.tif
MetaImage\24_ch_R_m3d_HE.tif
MetaImage\25_ch_R_mm3d_HE.tif
MetaImage\26_ch_R_mm3d_kuwahara_HE.tif
23_ch_R_fusion.tif
27_ch_R_HE_fusion.tif

--------------------------------------------------------------------------------
# RGB (10 files)

MetaImage\28_RGB_m3d.tif
MetaImage\29_RGB_mm3d.tif
MetaImage\30_RGB_mm3d_kuwahara.tif
MetaImage\33_RGB_m3d_HE.tif
MetaImage\34_RGB_mm3d_HE.tif
MetaImage\35_RGB_mm3d_kuwahara_HE.tif
31_RGB_fusion.tif
32_RGB_fusion2Gray.tif
36_RGB_HE_fusion.tif
37_RGB_HE_fusion2Gray.tif

--------------------------------------------------------------------------------
# ManualROI (at least 4 files)

ManualROI\ManualROI.roi
ManualROI\38_minBF_direct.tif
ManualROI\39_sumBF_direct.tif
ManualROI\40_stdBF_direct.tif
ManualROI\*.manualroi.tif ( if apply crop )

--------------------------------------------------------------------------------
# SLIC (5 files) [Repo: Zebrafish_Cell_Count]

SLIC\*_{dark_*}\*.ana.toml
SLIC\0*_{dark_*}\*.seg0.pkl
SLIC\*_{dark_*}\*.seg0.png
SLIC\*_{dark_*}\*.seg1.pkl
SLIC\*_{dark_*}\*.seg1.png
--------------------------------------------------------------------------------
```

## BrightField_analyze

```text
--------------------------------------------------------------------------------
# Auto (8 files)

MetaImage\00_original_16bit.tif
MetaImage\01_convert_8bit.tif
MetaImage\03_auto_threshold.tif
MetaImage\04_measured_mask.tif
MetaImage\RoiSet.roi
02_cropped_BF.tif
05_cropped_BF--MIX.tif
AutoAnalysis.csv

--------------------------------------------------------------------------------
# Manual (3 files)

ManualAnalysis.csv
Manual_cropped_BF--MIX.tif
Manual_measured_mask.tif

--------------------------------------------------------------------------------
# Unet Generated (3 files)

UNetAnalysis.csv
UNet_cropped_BF--MIX.tif [Repo: ZebraFish_BF_Seg]
UNet_predict_mask.tif [Repo: ZebraFish_BF_Seg]
--------------------------------------------------------------------------------
```
