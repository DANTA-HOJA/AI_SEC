import os
import sys
import re
from colorama import Fore, Back, Style
from pathlib import Path
import toml
import pandas as pd

abs_module_path = Path("./../../modules/").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from logger import init_logger
from db.SinglePredictionParser import SinglePredictionParser
from misc.CLIDivider import CLIDivider
cli_divider = CLIDivider()
cli_divider.process_start(use_tqdm=True)

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

prediction_dir_dict = {}
for prediction_dir in prediction_dir_list:
    name = str(prediction_dir).split(os.sep)[-1]
    name_split = re.split("{|}", name)
    if len(name_split) == 9:
        time_stamp = name_split[0]
        model_desc = name_split[5]
        key = f"{time_stamp}_{model_desc}"
        assert key not in prediction_dir_dict, (f"{Fore.RED}{Back.BLACK} Find multiple histories: time_stamp: '{time_stamp}', model_desc: '{model_desc}'. "
                                                f"Can not accept duplicate test conditions. {Style.RESET_ALL}\n")
        prediction_dir_dict.update({key: { "name": name, "path": prediction_dir}})

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

for key, value in prediction_dir_dict.items():
    
    name = value["name"]
    path = value["path"]
    
    single_pred_parser = SinglePredictionParser(path, log=log)
    
    if key in existing_history_dict:
        
        existing_name = existing_history_dict[key]
        
        if name == existing_name: pass
        else:
            log.info(f"{Fore.YELLOW} *** Update *** {Style.RESET_ALL}")
            parsed_df = single_pred_parser.parse()
            parsed_df.index = db_xlsx[(db_xlsx["History Name"] == existing_name)].index
            db_xlsx.loc[parsed_df.index] = parsed_df
            
        existing_history_dict.pop(key)
    
    else:
        log.info(f"{Fore.YELLOW} --- New History --- {Style.RESET_ALL}")
        parsed_df = single_pred_parser.parse()
        
        if db_xlsx is None:
            db_xlsx = parsed_df
        else:
            db_xlsx = pd.concat([db_xlsx, parsed_df], ignore_index=True)

# -----------------------------------------------------------------------------------
# Delete 'non-existing' history ( directory removed by the user )

for key in existing_history_dict.keys():
    
    existing_name = existing_history_dict[key]
    idx = db_xlsx[(db_xlsx["History Name"] == existing_name)].index
    db_xlsx = db_xlsx.drop(index=idx)
    
    log.info(f"{Fore.YELLOW}Delete : {Fore.MAGENTA} {existing_name} {Style.RESET_ALL}\n")

# -----------------------------------------------------------------------------------
# Sort and reset index

db_xlsx = db_xlsx.sort_values("History Name")
db_xlsx = db_xlsx.reset_index(drop=True)

# -----------------------------------------------------------------------------------
# Save `db_xlsx`

if db_xlsx_path.exists():
    
    if db_xlsx.equals(pd.read_excel(db_xlsx_path, engine="openpyxl", index_col=0)):
        log.info(f"{Fore.YELLOW} --- No Changed --- {Style.RESET_ALL}")
    else:
        print("Saving `db_xlsx`... ")
        db_xlsx.to_excel(db_xlsx_path)
        print(f"{Fore.GREEN}{Back.BLACK} Done! {Style.RESET_ALL}")

else:
    print("Saving `db_xlsx`... ")
    db_xlsx.to_excel(db_xlsx_path)
    print(f"{Fore.GREEN}{Back.BLACK} Done! {Style.RESET_ALL}")

# -----------------------------------------------------------------------------------
cli_divider.process_completed()