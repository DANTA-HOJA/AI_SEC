import os
import sys
import re
from colorama import Fore, Back, Style
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

model_prediction_root = db_root.joinpath(dbpp_config["model_prediction"])
prediction_dir_list = sorted(list(model_prediction_root.glob("*")), key=lambda x: str(x).split(os.sep)[-1])
# prediction_dir_list = prediction_dir_list[:2]

prediction_dir_dict = {str(prediction_dir).split(os.sep)[-1]: prediction_dir
                                        for prediction_dir in prediction_dir_list}

# rm "temp" and "un-tested" directory
for name in list(prediction_dir_dict.keys()):
    name_split = re.split("{|}", name)
    if len(name_split) <= 5: prediction_dir_dict.pop(name)

# -----------------------------------------------------------------------------------

db_xlsx_name = r"{DB}_Predictions"
db_xlsx_path = db_root.joinpath(f"{db_xlsx_name}.xlsx")
db_xlsx = None
existing_history_dict = {}

if db_xlsx_path.exists():
    
    db_xlsx = pd.read_excel(db_xlsx_path, engine="openpyxl", index_col=0)

    for name in list(db_xlsx["History Name"]):
        name_split = re.split("{|}", name)
        time_stamp = name_split[0]
        model_desc = name_split[5]
        key = f"{time_stamp}_{model_desc}"
        # update
        existing_history_dict.update({key : name})

# -----------------------------------------------------------------------------------

for name, path in prediction_dir_dict.items():
    
    name_split = re.split("{|}", name)
    time_stamp = name_split[0]
    model_desc = name_split[5]
    key = f"{time_stamp}_{model_desc}"
    
    single_pred_parser = SinglePredictionParser(path, log=log)
    
    if key in existing_history_dict:
        
        existing_name = existing_history_dict[key]
        
        if name == existing_name: pass
        else:
            log.info(f"{Fore.YELLOW} *** Update *** {Style.RESET_ALL}")
            parsed_df = single_pred_parser.parse()
            parsed_df.index = db_xlsx[(db_xlsx["History Name"] == existing_name)].index
            db_xlsx.loc[parsed_df.index] = parsed_df
    
    else:
        log.info(f"{Fore.YELLOW} --- New History --- {Style.RESET_ALL}")
        parsed_df = single_pred_parser.parse()
        
        if db_xlsx is None:
            db_xlsx = parsed_df
        else:
            db_xlsx = pd.concat([db_xlsx, parsed_df], ignore_index=True)

db_xlsx.to_excel(db_xlsx_path)