# result alias ↔ general name

## PalmSkin_preprocess

```text
"RGB_direct_max_zproj":          "*_RGB_direct_max_zproj.tif",
-----------------------------------------------------------------------------------
"ch_B":                          "MetaImage/*_B_processed.tif",
"ch_B_Kuwahara":                 f"MetaImage/*_B_processed_{Kuwahara}.tif",
"ch_B_fusion":                   "*_B_processed_fusion.tif",
"ch_B_HE":                       "MetaImage/*_B_processed_HE.tif",
"ch_B_Kuwahara_HE":              f"MetaImage/*_B_processed_{Kuwahara}_HE.tif",
"ch_B_HE_fusion":                "*_B_processed_HE_fusion.tif",
-----------------------------------------------------------------------------------
"ch_G":                          "MetaImage/*_G_processed.tif",
"ch_G_Kuwahara":                 f"MetaImage/*_G_processed_{Kuwahara}.tif",
"ch_G_fusion":                   "*_G_processed_fusion.tif",
"ch_G_HE":                       "MetaImage/*_G_processed_HE.tif",
"ch_G_Kuwahara_HE":              f"MetaImage/*_G_processed_{Kuwahara}_HE.tif",
"ch_G_HE_fusion":                "*_G_processed_HE_fusion.tif",
-----------------------------------------------------------------------------------
"ch_R":                          "MetaImage/*_R_processed.tif",
"ch_R_Kuwahara":                 f"MetaImage/*_R_processed_{Kuwahara}.tif",
"ch_R_fusion":                   "*_R_processed_fusion.tif",
"ch_R_HE":                       "MetaImage/*_R_processed_HE.tif",
"ch_R_Kuwahara_HE":              f"MetaImage/*_R_processed_{Kuwahara}_HE.tif",
"ch_R_HE_fusion":                "*_R_processed_HE_fusion.tif",
-----------------------------------------------------------------------------------
"RGB":                           "MetaImage/*_RGB_processed.tif",
"RGB_Kuwahara":                  f"MetaImage/*_RGB_processed_{Kuwahara}.tif",
"RGB_fusion":                    "*_RGB_processed_fusion.tif", => Average(RGB_processed, RGB_processed_Kuwahara)
"RGB_fusion2Gray":               "*_RGB_processed_fusion2Gray.tif",
"RGB_HE" :                       "MetaImage/*_RGB_processed_HE.tif",
"RGB_Kuwahara_HE" :              f"MetaImage/*_RGB_processed_{Kuwahara}_HE.tif",
"RGB_HE_fusion" :                "*_RGB_processed_HE_fusion.tif", => Average(RGB_processed_HE, "RGB_processed_Kuwahara_HE)
"RGB_HE_fusion2Gray":            "*_RGB_processed_HE_fusion2Gray.tif",
-----------------------------------------------------------------------------------
"BF_Zproj":                      f"MetaImage/*_{bf_zproj_type}.tif",
"BF_Zproj_HE":                   f"MetaImage/*_{bf_zproj_type}_HE.tif",
"Threshold":                     f"MetaImage/*_Threshold_{bf_treshold}.tif",
"outer_rect":                    "MetaImage/*_outer_rect.tif",
"inner_rect":                    "MetaImage/*_inner_rect.tif",
"RoiSet" :                       "MetaImage/RoiSet_AutoRect.zip",
-----------------------------------------------------------------------------------
"RGB_fusion--AutoRect":          "*_RGB_processed_fusion--AutoRect.tif",
"RGB_HE_fusion--AutoRect":       "*_RGB_processed_HE_fusion--AutoRect.tif",
-----------------------------------------------------------------------------------
"autocropped_RGB_fusion" :       "*_autocropped_RGB_processed_fusion.tif",
"autocropped_RGB_HE_fusion" :    "*_autocropped_RGB_processed_HE_fusion.tif",
```

## BrightField_analyze

```text
"original_16bit" :          "MetaImage/*_original_16bit.tif",
"cropped_BF" :              "*_cropped_BF.tif",
"AutoThreshold" :           f"MetaImage/*_AutoThreshold_{autothreshold_algo}.tif",
"measured_mask" :           "MetaImage/*_measured_mask.tif",
"cropped_BF--MIX" :         "*_cropped_BF--MIX.tif",
"RoiSet" :                  "MetaImage/RoiSet.zip",
"AutoAnalysis" :            "AutoAnalysis.csv",
"ManualAnalysis" :          "ManualAnalysis.csv",
"Manual_measured_mask" :    "Manual_measured_mask.tif",
"Manual_cropped_BF--MIX" :  "Manual_cropped_BF--MIX.tif",
```