import sys
from pathlib import Path
sys.path.append("./../modules/") # add path to scan customized module
from SurfaceAreaKMeansCluster import SurfaceAreaKMeansCluster

kmeans_rnd = 2022

# -----------------------------------------------------------------------------------
# xlsx: .../{Data}_Processed/{ Reminder }_Academia_Sinica_i[num]/data.xlsx

xlsx_path = Path( r"C:\Users\confocal_microscope\Desktop\WorkingDir\ZebraFish_DB\{Data}_Processed\{20230424_Update}_Academia_Sinica_i505\data.xlsx" )

# -----------------------------------------------------------------------------------
n_clusters = 3
label_str = ["S", "M", "L"]

# -----------------------------------------------------------------------------------
# n_clusters = 4
# label_str = ["S", "M", "L", "XL"]

# -----------------------------------------------------------------------------------
for log_scale in [False, True]:
    for kde in [False, True]:
        SAKMeansCluster = SurfaceAreaKMeansCluster(xlsx_path, n_clusters, label_str, kmeans_rnd,
                                                    log_base=10, cluster_with_log_scale=log_scale, with_kde=kde)
        print("="*80, "\n"); print(SAKMeansCluster); SAKMeansCluster.plot_and_save_xlsx()