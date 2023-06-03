import os
import sys
from pathlib import Path
import toml
import pandas as pd

rel_module_path = "./../../modules/"
sys.path.append( str(Path(rel_module_path).resolve()) ) # add path to scan customized module

from logger import init_logger
from db.SinglePredictionParser import SinglePredictionParser

config_dir = Path( "./../../Config/" ).resolve()

log = init_logger(f"Build DB Xlsx")

# -----------------------------------------------------------------------------------

with open(config_dir.joinpath("db_path_plan.toml"), mode="r") as f_reader:
    dbpp_config = toml.load(f_reader)
db_root = Path(dbpp_config["root"])

# -----------------------------------------------------------------------------------

predictions_dir_root = db_root.joinpath(dbpp_config["model_prediction"])
prediction_dir_list = sorted(list(predictions_dir_root.glob("*")), key=lambda x: str(x).split(os.sep)[-1])
# prediction_dir_list = prediction_dir_list[:2]

# rm "temp" directory
prediction_dir_dict = {str(prediction_dir).split(os.sep)[-1]: prediction_dir
                                        for prediction_dir in prediction_dir_list}
prediction_dir_dict.pop("temp")

# -----------------------------------------------------------------------------------

db_xlsx_name = r"{DB}_Predictions"
db_xlsx_path = db_root.joinpath(f"{db_xlsx_name}.xlsx")
db_xlsx = None

if db_xlsx_path.exists(): db_xlsx = pd.read_excel(db_xlsx_path, index_col=0)

# -----------------------------------------------------------------------------------

for prediction_dir in prediction_dir_dict.values():
    single_pred_parser = SinglePredictionParser(prediction_dir, log=log)
    parsed_df = single_pred_parser.parse()
    if db_xlsx is None:
        db_xlsx = parsed_df
    else:
        db_xlsx = pd.concat([db_xlsx, parsed_df], ignore_index=True)

db_xlsx.to_excel(db_xlsx_path)