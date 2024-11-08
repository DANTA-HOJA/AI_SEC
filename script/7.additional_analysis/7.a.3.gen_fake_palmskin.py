import os
import pickle as pkl
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import skimage as ski
from rich import print
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.dataset.dsname import get_dsname_sortinfo
from modules.data.dname import get_dname_sortinfo
from modules.data.processeddatainstance import ProcessedDataInstance
from modules.dl.fakepalmskin.utils import gen_singlecolor_palmskin
from modules.shared.clioutput import CLIOutput
from modules.shared.config import load_config
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import create_new_dir, get_repo_root

install()
# -----------------------------------------------------------------------------/


if __name__ == '__main__':
    
    print(f"Repository: '{get_repo_root()}'")
    
    """ Init components """
    processed_di = ProcessedDataInstance()
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
    dataset_base_size: str = "W512_H1024"
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
    
    """ Processed Data Instance """
    instance_desc = re.split("{|}", dataset_data)[1]
    temp_dict = {"data_processed": {"instance_desc": instance_desc}}
    processed_di.parse_config(temp_dict)
    csv_path = processed_di.instance_root.joinpath("data.csv")
    df: pd.DataFrame = pd.read_csv(csv_path, encoding='utf_8_sig')
    palmskin_dnames = sorted(pd.concat([df["Palmskin Anterior (SP8)"],
                                        df["Palmskin Posterior (SP8)"]]),
                            key=get_dname_sortinfo)
    
    """ Apply SLIC on each image """
    cli_out.divide()
    with Progress() as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(paths))
        
        for path, dname in zip(paths, palmskin_dnames):
            
            # check dsname, dname is match
            assert get_dsname_sortinfo(path) == get_dname_sortinfo(dname), "dsname, dname not match"
            
            result_name = path.stem
            dname_dir = path.parents[0]
            
            # check analyze condiction is same
            verify_cfg = dname_dir.joinpath(f"SLIC/{result_name}_{{dark_{dark}}}/{{copy}}_get_cell_feature.toml")
            assert config == load_config(verify_cfg), f"`verify_cfg` not match, '{verify_cfg}'"
            
            # load `patch_seg` (seg2)
            seg2_pkl = dname_dir.joinpath(f"SLIC/{result_name}_{{dark_{dark}}}/{result_name}.seg2.pkl")
            print(f"[ {seg2_pkl.parts[-1]} : '{seg2_pkl}' ]")
            with open(seg2_pkl, mode="rb") as f_reader:
                seg2 = pkl.load(f_reader)

            # generate fake palmskin images
            fake_tp1 = gen_singlecolor_palmskin(seg2, 0.0, 1.0, 0.0) # border white, cytosol black, and black background
            fake_tp2 = gen_singlecolor_palmskin(seg2, 0.5, 1.0, 0.0) # border white, cytosol gray, and black background
            fake_tp3 = gen_singlecolor_palmskin(seg2, 1.0, 1.0, 0.0) # border white, cytosol white, and black background

            """Save images"""
            dpath = processed_di.palmskin_processed_dname_dirs_dict[dname]
            fakeimg_dir = dpath.joinpath(f"FakeImage/{dataset_palmskin_result}_{{dark_{dark}}}")
            create_new_dir(fakeimg_dir)
            for enum, img in enumerate([fake_tp1, fake_tp2, fake_tp3], start=1):
                save_path = fakeimg_dir.joinpath(f"{dataset_palmskin_result}.faketype{enum}.tif")
                ski.io.imsave(save_path, np.uint8(img*255)) # float to 8-bit image
                print(f"Fake_Type{enum} : '{save_path}'")
            cli_out.new_line()
            
            # update pbar
            pbar.advance(task)
    
    cli_out.new_line()
    print("[green]Done! \n")
    # -------------------------------------------------------------------------/