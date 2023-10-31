import os
import sys
from glob import glob


import pandas as pd 



if __name__ == "__main__":
    
    print("processing...", "\n")
    
    
    ijm_results_path = r"C:\Users\confocal_microscope\Desktop\SA_SL_TEST\SA_SL_TEST--Result" # Scan all csv file contains SA_SL_result
    xlsx_out_path = "."
    
    
    csv_results_list = glob(f"{ijm_results_path}/*.csv")
    print(f"total csv found : {len(csv_results_list)}", "\n")
    
    
    df_xlsx = pd.DataFrame()
    for result in csv_results_list:
        df_csv_result = pd.read_csv(result)
        df_xlsx = pd.concat([df_xlsx, df_csv_result], ignore_index=True)
    
    
    df_xlsx = df_xlsx.drop(' ', axis=1) ## 刪除 imageJ 每筆 result 的自動編號
    print(df_xlsx.columns, "\n")
    
    
    # df_xlsx.loc["AVERAGE", "Area"] = df_xlsx["Area"].sum()/df_xlsx.shape[0] # 計算 Area 的平均
    print(df_xlsx)
    
    
    xlsx_out_path = os.path.join(xlsx_out_path, "SA_SL_result.xlsx")
    df_xlsx.to_excel(xlsx_out_path, engine="openpyxl")
    
    
    print("="*100, "\n", "process all complete !", "\n")