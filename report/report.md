# Log

## IMPORTANT

- 目前最大編號 : ```207```
- 最新 Data 資料夾 : ```{20221209_UPDATE_82}_Academia_Sinica_i324```

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

## 2023/01/26

- 修改測量 Surface area ( SA ), Standard length ( SL ) 的 ImageJ Macro

    1. 修正掃描 lif 檔時若資料夾內含有其他子資料夾會造成 Macro Error ( 無法開檔 )

    2. 修正 lif 檔 ( Leica microscope 格式，一個 lif 內有多條魚 ) 內只有 1 image 時存檔後的 tiff 只有 lif name 而不是 lif name + image name
        - e.g. 20220610_CE001_palmskin_8dpf --> 20220610_CE001_palmskin_8dpf - Series001 fish 1

    3. 修正 fish 出現兩種檔名的情況
        - fish 16 出現兩次 : "20220617_CE002_palmskin_8dpf - Series006 fish 16 palmskin_8dpf", "20220617_CE002_palmskin_8dpf - Series006 fish 16 palmskin_8dpf_000"

    4. 加入 "Find focused slices" 解決出現未對焦照片的問題
        - "20220617_CE002_palmskin_8dpf.lif" 裡出現 slices > 1 的狀況，且 slices 中只有一張有對焦，若沒有特別選擇都會直接拿第一張，但通常是第 4 或 5 張才有對焦
        - Plugin ref : <https://sites.google.com/site/qingzongtseng/find-focus>
        - Algorithm  : autofocus algorithm "Normalized variance"  (Groen et al., 1985; Yeo et al., 1993)

    5. 加入 "Set Scale" ，統一照片尺度不統一的問題
        - confocal 內部 meta data 記載 1 pixel = 6.5 micron, 換算後 0.3077 pixels/micron  

    6. 新增 Analysis 後，若找到的 ROI 不等於 1 ( ROI = 1 代表順利只抓到魚 ) 會在 Log file 標記 Error, 以便後續使用手動測量

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

## 2023/02/01

- ```fish_id = [4, 7, 68, 109, 110, 156]``` 因為 Bright Field + palmskin RGB 狀況不好予以刪除
- 新建 ```dataset_generate/``` 並移動 ```crop_img_A.py```, ```crop_img_P.py```, ```mk_dataset_horiz_cut.py``` 至其下
- 分離用來產生 dataset 的 subfunctions 至 ```dataset_generate_functions.py```
- 因應 新的 xlsx ( data.xlsx ) 欄位名稱，分別建立 ```dataset_generate/old_xlsx_col_name/```, ```dataset_generate/new_xlsx_col_name/``` 以利區分

## 2023/02/02

- 合併 ```dataset_generate/new_xlsx_col_name/``` 底下的 ```crop_img_A.py```、```crop_img_P.py```為一個檔案 ```mk_dataset_simple_crop.py``` ( 大調整 )

## 2023/02/03

- 嘗試以 "影像處理" 解決 Auto Fluorescence 未果，應該會採用 train 一個 Auto Fluorescence 的分類器
