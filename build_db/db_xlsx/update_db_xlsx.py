import os
import sys
from pathlib import Path
import pandas as pd

rel_module_path = "./../../modules/"
sys.path.append( str(Path(rel_module_path).resolve()) ) # add path to scan customized module

from logger import init_logger
from db.SinglePredictionParser import SinglePredictionParser

# -----------------------------------------------------------------------------------

zebrafish_db_root = Path(r"/home/rime97410000/ZebraFish_DB")
db_name = r"{DB}_Predictions"

# -----------------------------------------------------------------------------------
log = init_logger(f"SinglePredictionParser")

predictions_dir_root = zebrafish_db_root.joinpath(r"{Model}_Prediction")
prediction_dir_list = sorted(list(predictions_dir_root.glob("*")), key=lambda x: str(x).split(os.sep)[-1])
# prediction_dir_list = prediction_dir_list[:2]
num = 0
for prediction_dir in prediction_dir_list: # rm "temp"
    if str(prediction_dir_list[num]).split(os.sep)[-1] == "temp": prediction_dir_list.pop(num)
    else: num += 1

zebrafish_db = None
zebrafish_db_path = zebrafish_db_root.joinpath(f"{db_name}.xlsx")

# -----------------------------------------------------------------------------------
if zebrafish_db_path.exists(): zebrafish_db = pd.read_excel(zebrafish_db_path, index_col=0)


for prediction_dir in prediction_dir_list:
    single_pred_parser = SinglePredictionParser(prediction_dir, log=log)
    parsed_df = single_pred_parser.parse()
    if zebrafish_db is None:
        zebrafish_db = parsed_df
    else:
        zebrafish_db = pd.concat([zebrafish_db, parsed_df], ignore_index=True)


zebrafish_db.to_excel(zebrafish_db_path)