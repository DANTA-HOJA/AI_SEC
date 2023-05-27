import sys
from pathlib import Path
sys.path.append("./../modules/") # add path to scan customized module
from clustering.SurfaceAreaKMeansCluster import SurfaceAreaKMeansCluster

kmeans_rnd = 2022

# -----------------------------------------------------------------------------------
# xlsx: .../{Data}_Preprocessed/{ Reminder }_Academia_Sinica_i[num]/data.xlsx

xlsx_path = Path( r"C:\Users\confocal_microscope\Desktop\WorkingDir\ZebraFish_DB\{Data}_Preprocessed\{20230424_Update}_Academia_Sinica_i505\data.xlsx" )
old_classdiv_xlsx_path = Path( r"C:\Users\confocal_microscope\Desktop\WorkingDir\ZebraFish_DB\!~OLD_FILE\xlsx\!~BeforeCluster (20230508)\{20230424_Update}_Academia_Sinica_i505\{Modify}_xlsx\{3CLS_SURF_050STDEV}_data.xlsx" )

# -----------------------------------------------------------------------------------
n_clusters = 3
label_str = ["S", "M", "L"]

# -----------------------------------------------------------------------------------
# n_clusters = 4
# label_str = ["S", "M", "L", "XL"]

# -----------------------------------------------------------------------------------
for arg_6 in [False, True]:
    for arg_7 in [False, True]: # arg_7: x_axis_log_scale = True，產生的 image 會有 kde curve（比較好區分 "原始刻度" 和 "LOG" 刻度）
        SAKMeansCluster = SurfaceAreaKMeansCluster(xlsx_path, n_clusters, label_str, kmeans_rnd,
                                                   log_base=10, cluster_with_log_scale=arg_6, x_axis_log_scale=arg_7,
                                                   old_classdiv_xlsx_path=old_classdiv_xlsx_path)
        print("="*80, "\n"); print(SAKMeansCluster); SAKMeansCluster.plot_and_save_xlsx()