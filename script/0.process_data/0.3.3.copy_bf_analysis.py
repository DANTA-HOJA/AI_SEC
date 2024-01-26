import os
import shutil
import sys
from pathlib import Path

from rich import print
from rich.progress import *

abs_module_path = Path("./../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.utils import (create_new_dir, get_coupled_config_name,
                                  get_repo_root)
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

# set variables
cli_out = CLIOutput()
cli_out._set_logger("Copy Brightfield Analysis")
config = load_config(get_coupled_config_name(__file__))
old_ins: str = config["instance_desc"]["old"]
new_ins: str = config["instance_desc"]["new"]
image_type: str = config["copy_info"]["image_type"]
result_names: list[str] = config["copy_info"]["result_name"]

""" Old Instance ( with manual analysis results ) """
old_di = ProcessedDataInstance()
old_di._cli_out._set_logger("ProcessedDataInstance (OLD)")
old_di.parse_config({"data_processed": {"instance_desc": old_ins}})
old_processed_dir: Path = getattr(old_di, f"{image_type}_processed_dir")

""" New Instance """
new_di = ProcessedDataInstance()
new_di._cli_out._set_logger("ProcessedDataInstance (NEW)")
new_di.parse_config({"data_processed": {"instance_desc": new_ins}})
new_processed_dir: Path = getattr(new_di, f"{image_type}_processed_dir")

""" Main Process """
progress = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TextColumn("{task.completed} of {task.total}"),
    auto_refresh=False
)

cli_out.divide()
with progress:

    skipped_result: str = ""
    for result_name in result_names:
        
        # >>> old target results <<<
        rel_path, old_dict = \
            old_di.get_sorted_results_dict(image_type, result_name,
                                           allow_glob_dir=True)
        if len(old_dict) == 0:
            skipped_result += f"[magenta]Can't find any '{result_name}'\n"
            continue
        
        # >>> new target results <<<
        _, new_dict = \
            new_di.get_sorted_results_dict(image_type, result_name,
                                           allow_glob_dir=True)
        
        task_desc = f"[yellow]{cli_out.logger_name}: '{result_name}'..."
        task = progress.add_task(task_desc, total=len(old_dict))
        
        for fish_dname, old_path in old_dict.items():
            
            try:
                # >>> `new_di` has target file? <<<
                new_dict.pop(fish_dname)
                
            except KeyError:
                
                dname_dir = new_processed_dir.joinpath(fish_dname)
                if dname_dir.exists():
                    # >>> fish (dname) exists <<<
                    new_path = dname_dir.joinpath(rel_path)
                    
                    if old_path.is_dir():
                        shutil.copytree(old_path, new_path)
                        
                    elif old_path.is_file():
                        create_new_dir(os.path.split(new_path)[0])
                        shutil.copy(old_path, new_path)
                    
                    print(f"[#2596be]'{old_path}'\n [#FFFFFF]--> [#be4d25]'{new_path}'\n")
                
            progress.update(task, advance=1)
            progress.refresh()

cli_out.divide()
print(skipped_result)