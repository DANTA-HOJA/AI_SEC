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
from modules.shared.utils import get_coupled_config_name, get_repo_root
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

# set variables
cli_out = CLIOutput()
cli_out._set_logger("Copy Brightfield Analysis")
config = load_config(get_coupled_config_name(__file__))
old_ins = config["instance_desc"]["old"]
new_ins = config["instance_desc"]["new"]
filenames = config["copy_info"]["filenames"]

""" Old Instance ( with manual analysis results ) """
old_di = ProcessedDataInstance()
old_di._cli_out._set_logger("ProcessedDataInstance (OLD)")
old_di.parse_config({"data_processed": {"instance_desc": old_ins}})
old_dict = old_di.brightfield_processed_dname_dirs_dict

""" New Instance """
new_di = ProcessedDataInstance()
new_di._cli_out._set_logger("ProcessedDataInstance (NEW)")
new_di.parse_config({"data_processed": {"instance_desc": new_ins}})
new_dict = new_di.brightfield_processed_dname_dirs_dict


""" Main Process """
progress = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TextColumn("{task.completed} of {task.total}"),
    auto_refresh=False
)

cli_out.divide()
with progress:
    
    task_desc = f"[yellow]{cli_out.logger_name}..."
    task = progress.add_task(task_desc, total=len(old_dict))
    
    for dname, old_dir in old_dict.items():
        
        try:
            new_dir = new_dict[dname]
            for filename in filenames:
                
                old_path = old_dir.joinpath(filename)
                new_path = new_dir.joinpath(filename)
                
                if old_path.exists():
                    shutil.copy(old_path, new_path)
                    print(f"[#2596be]'{old_path}'\n [#FFFFFF]--> [#be4d25]'{new_path}'\n")
            
            progress.update(task, advance=1)
            progress.refresh()
            
        except KeyError:
            pass

cli_out.new_line()