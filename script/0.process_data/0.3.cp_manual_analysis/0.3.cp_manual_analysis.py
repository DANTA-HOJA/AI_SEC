import sys
from pathlib import Path
from colorama import Fore, Back, Style
import shutil

abs_module_path = Path("./../../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.clioutput import CLIOutput
from modules.shared.utils import get_repo_root

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

""" CLI output """
cli_out = CLIOutput()

""" Old Instance ( with manual analysis results ) """
cli_out.divide()
old_instance = ProcessedDataInstance()
old_instance.load_config(Path("old_instance.toml"))
old_dict = old_instance.brightfield_processed_dname_dirs_dict

""" New Instance """
cli_out.divide()
new_instance = ProcessedDataInstance()
new_instance.load_config(Path("new_instance.toml"))
new_dict = new_instance.brightfield_processed_dname_dirs_dict


""" Main Process """
cli_out.divide()
filenames = ["Manual_cropped_BF--MIX.tif", "Manual_measured_mask.tif", "ManualAnalysis.csv"]

for key, value in new_dict.items():
    
    for filename in filenames:
        
        new_path = value.joinpath(filename)
        old_path = old_dict[key].joinpath(filename)
        
        if not new_path.exists():
            shutil.copy(old_path, new_path)
            print(f"{Fore.BLUE}'{old_path}'\n -> {Fore.YELLOW}'{new_path}'{Style.RESET_ALL}\n")