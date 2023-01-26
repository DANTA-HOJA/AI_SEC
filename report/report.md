2023/01/26

- 修改測量 Surface area ( SA ), Standard length ( SL ) 的 ImageJ Macro
    
    1. 修正掃描 lif 檔時若資料夾內含有其他子資料夾會造成 Macro Error ( 無法開檔 )

    2. 修正 lif 檔 ( Leica microscope 格式，一個 lif 內有多條魚 ) 內只有 1 image 時存檔後的 tiff 只有 lif name 而不是 lif name + image name
        - e.g. 20220610_CE001_palmskin_8dpf --> 20220610_CE001_palmskin_8dpf - Series001 fish 1

    3. 修正 fish 出現兩種檔名的情況
        - fish 16 出現兩次 : "20220617_CE002_palmskin_8dpf - Series006 fish 16 palmskin_8dpf", "20220617_CE002_palmskin_8dpf - Series006 fish 16 palmskin_8dpf_000"
    
    4. 加入 "Find focused slices" 解決出現未對焦照片的問題
        - "20220617_CE002_palmskin_8dpf.lif" 裡出現 slices > 1 的狀況，且 slices 中只有一張有對焦，若沒有特別選擇都會直接拿第一張，但通常是第 4 或 5 張才有對焦
        - Plugin ref : https://sites.google.com/site/qingzongtseng/find-focus
		- Algorithm  : autofocus algorithm "Normalized variance"  (Groen et al., 1985; Yeo et al., 1993)

    5. 加入 "Set Scale" ，統一照片尺度不統一的問題
        - confocal 內部 meta data 記載 1 pixel = 6.5 micron, 換算後 0.3077 pixels/micron  
    
    6. 新增 Analysis 後，若找到的 ROI 不等於 1 ( ROI = 1 代表順利只抓到魚 ) 會在 Log file 標記 Error, 以便後續使用手動測量

    7. 檔名優化
        | functional | file name |
        | :----: | :----: |
        | 從 lif 中分離的每條魚                                | 20220610_CE001_palmskin_8dpf - Series001 fish 1.tif |
        | Cropped 主要區域, postfix "_Cropped"                | 20220610_CE001_palmskin_8dpf - Series001 fish 1_Cropped.tif |
        | Threshold, postfix "_Threshold"                    | 20220610_CE001_palmskin_8dpf - Series001 fish 1_Threshold.tif |
        | Analysis 後產生的 Mask, postfix "--Mask"            | 20220610_CE001_palmskin_8dpf - Series001 fish 1--Mask.tif |
        | Cropped 和 Mask 做 AND 運算後的結果, postfix "--MIX" | 20220610_CE001_palmskin_8dpf - Series001 fish 1--MIX.tif |
        | Analysis 產生的 CSV (自動)                          | 20220610_CE001_palmskin_8dpf - Series001 fish 1_AutoAnalysis.csv |
        | Analysis 產生的 CSV (自動失敗 -> 手動)               | 20220610_CE001_palmskin_8dpf - Series001 fish_Manual.csv |




    