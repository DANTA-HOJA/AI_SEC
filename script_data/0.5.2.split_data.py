import sys
from pathlib import Path

import pandas as pd
from rich import print
from rich.panel import Panel
from rich.pretty import Pretty

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config, dump_config
from modules.shared.utils import get_repo_root
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

training_ratio = 0.8
train_ratio = 0.9

# set variables
cli_out = CLIOutput()
cli_out._set_logger("Split Dataset")
config = load_config("0.5.cluster_data.toml")
random_seed = config["cluster"]["random_seed"]

""" Read `data.csv` """
processed_di = ProcessedDataInstance()
processed_di.parse_config(config)
df = pd.read_csv(processed_di.tabular_file, encoding='utf_8_sig', index_col=[0])
cli_out.divide()

""" Main Process """
training_df: pd.DataFrame = df.sample(frac=training_ratio, replace=False, random_state=random_seed)
test_df: pd.DataFrame = df[~df.index.isin(training_df.index)]

train_df: pd.DataFrame = training_df.sample(frac=train_ratio, replace=False, random_state=random_seed)
valid_df: pd.DataFrame = training_df[~training_df.index.isin(train_df.index)]

# display
tmp_dict = {}
tmp_dict["original_df"] = len(df)
tmp_dict["training_df"] = len(training_df)
tmp_dict["test_df"] = len(test_df)
tmp_dict["train_df"] = len(train_df)
tmp_dict["valid_df"] = len(valid_df)
print(Panel(Pretty(tmp_dict, expand_all=True), width=100))
dump_config(processed_di.instance_root.joinpath("split_count.log"), tmp_dict)

# apply split result to `df`
for idx in train_df.index:
    df.loc[idx, "dataset"] = "train"

for idx in valid_df.index:
    df.loc[idx, "dataset"] = "valid"

for idx in test_df.index:
    df.loc[idx, "dataset"] = "test"

# save file
df.to_csv(processed_di.instance_root.joinpath(f"datasplit_{random_seed}.csv"), encoding='utf_8_sig')
cli_out.new_line()