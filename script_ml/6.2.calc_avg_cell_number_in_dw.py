import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.dataset.dsname import get_dsname_sortinfo
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import get_repo_root

install()
# -----------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")

    """ Init components """
    path_navigator = PathNavigator()
    cli_out = CLIOutput()
    cli_out.divide()

    # load config
    # `dark` and `merge` are two parameters as color space distance, determined by experiences
    config = load_config("get_cell_feature.toml")
    # [dataset]
    dataset_seed_dir: str = config["dataset"]["seed_dir"]
    dataset_data: str = config["dataset"]["data"]
    dataset_palmskin_result: str = config["dataset"]["palmskin_result"]
    dataset_base_size: str = config["dataset"]["base_size"]
    # [SLIC]
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    merge: int       = config["SLIC"]["merge"]
    debug_mode: bool = config["SLIC"]["debug_mode"]
    print("", Pretty(config, expand_all=True))
    cli_out.divide()

    """ Colloct image file names """
    dataset_cropped: Path = path_navigator.dbpp.get_one_of_dbpp_roots("dataset_cropped_v3")
    src_root = dataset_cropped.joinpath(dataset_seed_dir,
                                        dataset_data,
                                        dataset_palmskin_result,
                                        dataset_base_size)
    paths = sorted(src_root.glob("*/*/*.tiff"), key=get_dsname_sortinfo)
    print(f"Total files: {len(paths)}")

    # variable
    cell_number: list = []
    
    """ Read analysis file """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(paths))
        
        for path in paths:
            
            result_name = path.stem
            dname_dir = path.parents[0]
            
            # check analyze condiction is same
            verify_cfg = dname_dir.joinpath(f"SLIC/{result_name}_{{dark_{dark}}}/{{copy}}_get_cell_feature.toml")
            assert config == load_config(verify_cfg), f"`verify_cfg` not match, '{verify_cfg}'"
            
            ana_file = dname_dir.joinpath(f"SLIC/{result_name}_{{dark_{dark}}}/{result_name}.ana.toml")
            print(f"[ {dname_dir.parts[-1]} : '{ana_file}' ]")
            analysis_dict = load_config(ana_file)
            cell_number.append(analysis_dict["cell_count"])
            
            # update pbar
            pbar.advance(task)
    
    """ Get Q2 cell number """
    print("", Pretty(config, expand_all=True))
    cli_out.divide()
    print(f"Q2 Cell Number = {np.quantile(cell_number, 0.5)}")
    print(f"Avg Cell Number = {np.mean(cell_number)} ± {np.std(cell_number)}")
    print(f"(round) Avg Cell Number = {round(np.mean(cell_number), 1)} ± {round(np.std(cell_number), 1)}")
    
    sns.displot(cell_number)
    plt.show()
    
    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/