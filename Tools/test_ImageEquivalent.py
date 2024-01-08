import os
import re
import sys
from pathlib import Path

import cv2
import numpy
from rich import print
from rich.pretty import Pretty
from rich.progress import *

abs_module_path = Path("./../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.utils import get_coupled_config_name
# -----------------------------------------------------------------------------/

# set variables
cliout = CLIOutput("Image Equivalent")
config = load_config(get_coupled_config_name(__file__))
old_ins = config["instance_desc"]["old"]
new_ins = config["instance_desc"]["new"]
image_type = config["image_info"]["image_type"]
result_file = config["image_info"]["result_file"]

# init `ProcessedDataInstance`
old_di = ProcessedDataInstance()
old_di._cli_out._set_logger("ProcessedDataInstance (OLD)")
old_di.parse_config({"data_processed": {"instance_desc": old_ins}})
new_di = ProcessedDataInstance()
new_di._cli_out._set_logger("ProcessedDataInstance (NEW)")
new_di.parse_config({"data_processed": {"instance_desc": new_ins}})
cliout.divide()

# get `paths` in `ProcessedDataInstance`
_, old_paths = old_di.get_sorted_results(image_type, result_file)
cliout.write(f"old: {old_di.instance_name}, "
             f"detect {len(old_paths)} files")
_, new_paths = new_di.get_sorted_results(image_type, result_file)
cliout.write(f"new: {new_di.instance_name}, "
             f"detect {len(new_paths)} files")
cliout.divide()

# get longer list
longer = 0
if len(old_paths) > len(new_paths):
    longer = old_paths
else:
    longer = new_paths

# start compare
final_path = []
progress = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TextColumn("{task.completed} of {task.total}")
)

with progress:

    task1 = progress.add_task("[red]Image Equivalent...", total=len(longer))
    
    for old_path, new_path in zip(old_paths, new_paths):
        
        img1 = cv2.imread(str(old_path))
        img2 = cv2.imread(str(new_path))
        final_path = [old_path, new_path]
        
        if not numpy.array_equal(img1, img2):
            cliout.divide()
            print("ERROR: ", Pretty(final_path, expand_all=True))
            raise ValueError()
        
        progress.update(task1, advance=1)

cliout.divide()
print("Last Path: ", Pretty(final_path, expand_all=True))
cliout.new_line()