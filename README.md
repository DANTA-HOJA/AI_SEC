# ZebraFish_AP_POS

MD705 cooperation project ( zebrafish size classifier by ```Anterior```, ```Posterior``` )

## Data Information ( *temporarily halt updates* )

- 目前最大編號 : ```207```
- 作為統計依據的 Data 資料夾 : ```{20221209_UPDATE_82}_Academia_Sinica_i324```

- 缺號統計 ( 共 45 筆 ) :
  - BrightField ( 共 5 筆 ) : ```[ 79, 87, 89, 93, 185 ]```
  - stacked_palmskin_RGB ( 共 40 筆 ) :

    - 20220613_CE001_palmskin_11dpf : ```[ 9, 10 ]``` (2)
    - 20220617_CE002_palmskin_8dpf : ```[ 12 ]``` (1)
    - 20220621_CE003_palmskin_8dpf : ```[ 24, 27, 28 ]``` (3)
    - 20220623_CE004_palmskin_7dpf : ```[ 35 ]``` (1)
    - 20220627_CE006_palmskin_8dpf : ```[ 57, 58, 59, 61, 62, 63, 64, 65 ]``` (8)
    - 20220628_CE006_palmskin_8dpf : ```[ 66, 67, 69, 70, 73 ]``` (5)
    - 20220708_CE009_palmskin_8dpf : ```[ 75, 76, 78, (BF_79), 80, 82, 83, 84 ]``` (8)
    - 20220711_CE009_palmskin_11dpf : ```[ (BF_87), 88, (BF_89) ]``` (3)
    - 20220719_CE011_palmskin_8dpf : ```[ (BF_93), 94, 97, 100 ]``` (4)
    - 20220727_CE012_palmskin_9dpf : ```[ 117 ]``` (1)
    - 20220816_AI001_palmskin_8dpf : ```[ 118 ]``` (1)
    - 20220823_AI002_palmskin_8dpf : ```[ 122 ]``` (1)
    - 20220825_AI002_palmskin_10dpf : ```[ 128 ]``` (1)
    - 20220901_AI003_palmskin_10dpf : ```[ 148, 149, 153 ]``` (3)
    - 20221014_AI004_palmskin_11dpf : ```[ 162 ]``` (1)
    - 20221125_AI005_palmskin_10dpf : ```[ 179, (BF_185) ]``` (2)

- ```[4, 7, 68, 109, 110, 156]``` 因為 Bright Field + palmskin RGB ( Auto Fluorescence ) 狀況不好予以刪除
- palmskin_raw_lif : ```[ Ch1, Ch2, Ch3, Ch4 ] -> [ B, G, R, (BF) ]```

## Log ( by date )

### 2023/01/18

- 中研院 *ImageJ Macro* 原始檔名 ： *20220614 macro for SL and SA measurement_by KY.ijm* ，用於計算 Surface area ( *SA* ), Standard length ( *SL* ) ( under ```{NAS_DL}_Academia_Sinica_Data/``` )
- 修改+優化 *ImageJ Macro*，可開始自行運算 *SA*, *SL* ，不需依賴中研院提供的 ```Machine learning.xlsx```，且改由 ```data_operate/``` 底下一系列的 *new scripts* 產生 ```XLSX_FILE``` 並更名為 ```data.xlsx```
- 修正後 ImageJ Macro 更名為 : ```[20230118_mod] macro for SL and SA measurement by SRY.ijm```

### 2023/01/26

- [ImageJ Macro](/data_operate/imagej_macro/%5B20230118_mod%5D%20macro%20for%20SL%20and%20SA%20measurement%20by%20SRY.ijm) 的修改細項 :

    1. 修正掃描 ```LIF_FILE``` 時若資料夾內含有其他子資料夾會造成 *Macro Error* ( 無法開檔 )

    2. 修正 ```LIF_FILE``` ( Leica microscope 格式；一個 lif 內可以有多條魚 ) 內只有 1 image 時，存檔後的 tiff 只有 *lif_name* 而不是 *lif_name + image_name*
        - e.g. *20220610_CE001_palmskin_8dpf* --> *20220610_CE001_palmskin_8dpf - Series001 fish 1*

    3. 修正同一條魚出現兩種檔名的情況
        - fish 16 出現兩次 : *20220617_CE002_palmskin_8dpf - Series006 fish 16 palmskin_8dpf*, *20220617_CE002_palmskin_8dpf - Series006 fish 16 palmskin_8dpf_000*

    4. 加入 *plugin* : ```Find focused slices``` 解決出現未對焦照片的問題 ( unpack 之後魚的 ```Z axis > 1```（仍為 Stack）)
        - *20220617_CE002_palmskin_8dpf.lif* 裡出現 ```slices > 1``` 的狀況，且 ```slices``` 中只有一張有對焦，若沒有特別選擇都會直接拿第一張，但通常是第 4 或 5 張才有對焦
        - Plugin ref : <https://sites.google.com/site/qingzongtseng/find-focus>
        - Algorithm  : autofocus algorithm *"Normalized variance"*  (Groen et al., 1985; Yeo et al., 1993)

    5. 加入 *ij_cmd* : ```Set Scale``` ，統一照片尺度不統一的問題
        - confocal 內部 meta_data 記載 *1 pixel = 6.5 micron*, 換算後 *0.3077 pixels/micron*

    6. 新增 *function* : Analysis 後，若找到的 ```ROI != 1``` ( ```ROI == 1``` 代表順利只抓到魚 ) 會在 *Log* 標記 *Error*, 以便後續手動測量

    7. 檔名優化
        | Function | File name |
        | :---- | :---- |
        | 從 lif 中分離的每條魚                                | 20220610_CE001_palmskin_8dpf - Series001 fish 1.tif |
        | Cropped 主要區域, postfix "_Cropped"                | 20220610_CE001_palmskin_8dpf - Series001 fish 1_Cropped.tif |
        | Threshold, postfix "_Threshold"                    | 20220610_CE001_palmskin_8dpf - Series001 fish 1_Threshold.tif |
        | Analysis 後產生的 Mask, postfix "--Mask"            | 20220610_CE001_palmskin_8dpf - Series001 fish 1--Mask.tif |
        | Cropped 和 Mask 做 AND 運算後的結果, postfix "--MIX" | 20220610_CE001_palmskin_8dpf - Series001 fish 1--MIX.tif |
        | Analysis 產生的 CSV (自動)                          | 20220610_CE001_palmskin_8dpf - Series001 fish 1_AutoAnalysis.csv |
        | Analysis 產生的 CSV (自動失敗 -> 手動)               | 20220610_CE001_palmskin_8dpf - Series001 fish_Manual.csv |

### 2023/02/01

- ```fish_id = [4, 7, 68, 109, 110, 156]``` 因為 Bright Field + palmskin RGB 狀況不好予以刪除
- 新建 ```dataset_generate/``` 並移動 ```crop_img_A.py```, ```crop_img_P.py```, ```mk_dataset_horiz_cut.py``` 至該資料夾底下
- 分離用來產生 dataset 的 subfunctions 至 ```dataset_generate_functions.py```
- 因應 *new_column_name* in ```data.xlsx```  欄位名稱，分別建立 ```dataset_generate/old_xlsx_col_name/```, ```dataset_generate/new_xlsx_col_name/``` 以利區分

### 2023/02/02

- 合併 ```dataset_generate/new_xlsx_col_name/``` 底下的 ```crop_img_A.py```、```crop_img_P.py```為一個檔案 ```mk_dataset_simple_crop.py``` ( 大調整 )

### 2023/02/03

- ( ***TODO*** ) 嘗試以 "影像處理" 解決 Auto Fluorescence 未果，應該會採用 train 一個 Auto Fluorescence 的分類器

### 2023/02/04

- 將 ```dataset_generate/new_xlsx_col_name/``` 底下的 ```mk_dataset_simple_crop.py``` 修改成會儲存 drop_images ( 有效資訊量太少的 images )

  - 在 ```Dataset_Name/[test, train]``` 下開始區分 ```selected```, ```drop``` 之後才是 ```FISH_SIZE``` ， *Log* 位置不變

### 2023/02/10

- *2023/02/05* - *2023/02/10* 為了重新處理 palmskin_RGB 雜訊問題，研究如何使用 PyImageJ，希望只留下 image processing 給 ImageJ，其他 (例如: 找檔案、修改檔名... ) 使用 Python 解決

  - *2023/02/05* - *2023/02/09* 跟著 [Tutorial](https://pyimagej.readthedocs.io/en/latest/index.html) 操作 ( [Try_PyImageJ](/FuncTest_with_ipynb/Try_PyImageJ.ipynb) )
  - *2023/02/10* 成功在 Python 中呼叫 ```Bio-Formats Plugin``` 且可偵測 ```LIF_FILE``` 內有幾張照片

### 2023/02/12

- 成功使用 ```PyImageJ``` ( require ```jpype```, ```scyjava``` ) 改寫 [ImageJ Macro](/data_operate/imagej_macro/%5B20230118_mod%5D%20macro%20for%20SL%20and%20SA%20measurement%20by%20SRY.ijm) 為 [imagej_BF_Analysis](/data_operate/BrightField/imagej_BF_Analysis.ipynb)

### 2023/02/13

- 調整 NAS 上 *BrightField_raw_lif* 、 *palmskin_raw_lif* 內的檔案名稱，確保兩者檔名相似性 ( 只有部分分隔符號不同 )
- 調整 NAS 上的資料夾名稱 ( 新舊資料夾名稱比對 : [圖片](/(doc)_pngs/OldNewDirNameCompare.png) )

### 2023/02/14

- 調整 ```BF_Analysis``` 下的資料夾命名，不再複製上級資料夾名稱，直接以 ```TIFF```, ```MetaImage```, ```Result``` 作為子資料夾名稱
- ```BF_Analysis``` 不再額外複製 ```{NAS_DL}_*/.../BrightField_RAW/``` ，改採 *直接讀取* 檔案產生 *SA*, *SL* 計算結果
- ```BF_Analysis``` 開始加上 prefix， {} 內用於紀錄該次針對計算所調整的項目
- 合併 ```/data_operate/BrightField/collect_BF_raw_lif.py``` 的操作合併至 [imagej_BF_Analysis](/data_operate/BrightField/imagej_BF_Analysis.ipynb)
