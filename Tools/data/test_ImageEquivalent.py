import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from rich import print
from rich.pretty import Pretty
from rich.progress import *

abs_module_path = Path("./../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.assert_fn import assert_file_ext
from modules.data import dname
from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.clioutput import CLIOutput
from modules.shared.config import get_coupled_config_name, load_config
# -----------------------------------------------------------------------------/

# set variables
cli_out = CLIOutput()
cli_out._set_logger("Image Equivalent")
config = load_config(get_coupled_config_name(__file__))
print(Pretty(config, expand_all=True))

old_ins = config["instance_desc"]["old"]
new_ins = config["instance_desc"]["new"]
image_type = config["image_info"]["image_type"]
result_name = config["image_info"]["result_name"]
assert_file_ext(result_name, ".tif")

# init `ProcessedDataInstance`
old_di = ProcessedDataInstance()
old_di._cli_out._set_logger("ProcessedDataInstance (OLD)")
old_di.parse_config({"data_processed": {"instance_desc": old_ins}})
new_di = ProcessedDataInstance()
new_di._cli_out._set_logger("ProcessedDataInstance (NEW)")
new_di.parse_config({"data_processed": {"instance_desc": new_ins}})
cli_out.divide()

# get `paths` in `ProcessedDataInstance`
_, old_paths = old_di.get_sorted_results_dict(image_type, result_name)
old_paths = list(old_paths.values())
cli_out.write(f"old: {old_di.instance_name}, "
              f"detect {len(old_paths)} files")
_, new_paths = new_di.get_sorted_results_dict(image_type, result_name)
new_paths = list(new_paths.values())
cli_out.write(f"new: {new_di.instance_name}, "
              f"detect {len(new_paths)} files")

# get longer list
longer = 0
if len(old_paths) > len(new_paths):
    longer = old_paths
else:
    longer = new_paths

# start compare
equal_interval = {"start": None, "end": None}
progress = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TextColumn("{task.completed} of {task.total}"),
    auto_refresh=False
)

cli_out.divide()
with progress:

    task_desc = f"[yellow]{cli_out.logger_name}..."
    task = progress.add_task(task_desc, total=len(longer))
    
    for _ in range(len(longer)):
            
        if len(old_paths) > 0:
            old_info = dname.get_dname_sortinfo(old_paths[0])
            tmp = 1 if old_info[1] == "A" else 2
            old_id = old_info[0]*10 + tmp
        else: old_id = np.inf
        
        if len(new_paths) > 0:
            new_info = dname.get_dname_sortinfo(new_paths[0])
            tmp = 1 if new_info[1] == "A" else 2
            new_id = new_info[0]*10 + tmp
        else: new_id = np.inf
        
        if old_id == new_id:
            
            old_path = old_paths.pop(0)
            new_path = new_paths.pop(0)
            broken_img = False
            
            img1 = cv2.imread(str(old_path))
            if img1 is None:
                broken_img = True
                print(f"Fish {old_info} is broken: [red]'{old_path}'\n")
            
            img2 = cv2.imread(str(new_path))
            if img2 is None:
                broken_img = True
                print(f"Fish {new_info} is broken: [red]'{new_path}'\n")
            
            if broken_img is True:
                progress.update(task, advance=1)
                progress.refresh()
                continue
            
            if equal_interval["start"] is None:
                equal_interval["start"] = old_info
            else:
                equal_interval["end"] = old_info
            
            if not np.array_equal(img1, img2):
                cli_out.divide()
                print("ERROR: ", Pretty([old_path, new_path], expand_all=True))
                raise ValueError
        
        else:
            if old_id < new_id:
                old_paths.pop(0)
                print(f"Fish {old_info} skipped: One of image is missing")
            else:
                new_paths.pop(0)
                print(f"Fish {new_info} skipped: One of image is missing")
        
        progress.update(task, advance=1)
        progress.refresh()

cli_out.divide()
print("Equal Interval: ", Pretty(equal_interval))
cli_out.new_line()