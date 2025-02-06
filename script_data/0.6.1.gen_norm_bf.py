import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import skimage as ski
from rich import print
from rich.console import Console
from rich.pretty import Pretty
from rich.progress import track
from rich.traceback import install

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.utils import get_repo_root

install()
# -----------------------------------------------------------------------------/

def gen_fitted_bf(fish_dname, masks, bfs):
    """
    """
    mask = cv2.imread(str(masks[fish_dname]), -1)
    bf = cv2.imread(str(bfs[fish_dname]), -1)
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return bf[y:y+h, x:x+w], mask[y:y+h, x:x+w]
    # -------------------------------------------------------------------------/


if __name__ == '__main__':
    
    print(f"Repository: '{get_repo_root()}'")
    
    # init components
    console = Console()
    processed_di = ProcessedDataInstance()
    
    # read df
    processed_di.parse_config("0.3.1.analyze_brightfield.toml")
    csv_path = processed_di.instance_root.joinpath("data.csv")
    df: pd.DataFrame = pd.read_csv(csv_path, encoding='utf_8_sig')
    
    # find `median` area in BFs
    median_area = df["Trunk surface area, SA (um2)"].median()
    tmp_row = df[df["Trunk surface area, SA (um2)"] == median_area]
    median_fish = list(tmp_row["Brightfield"])[0]
    
    # get masks (merge `manual` and `unet`)
    _, masks = processed_di.get_sorted_results_dict("brightfield", "UNet_predict_mask.tif")
    _, manual_masks = processed_di.get_sorted_results_dict("brightfield", "Manual_measured_mask.tif")
    masks.update(manual_masks)
    
    # get BFs
    _, bfs = processed_di.get_sorted_results_dict("brightfield", "02_cropped_BF.tif")
    target_fitted_bf, _ = gen_fitted_bf(median_fish, masks, bfs)
    
    for dname in track(df["Brightfield"], transient=True,
                       description=f"[cyan]Processing... ",
                       console=console):
        
        fitted_bf, fitted_mask = gen_fitted_bf(dname, masks, bfs)
        
        # save normalized BF
        norm_bf = cv2.resize(fitted_bf, target_fitted_bf.shape[::-1],
                             interpolation=cv2.INTER_LANCZOS4)
        save_path = bfs[dname].parent.joinpath("Norm_BF.tif")
        ski.io.imsave(save_path, norm_bf)
        console.print(f"{dname} : '{save_path}'")
        
        # save normalized Mask
        norm_mask = cv2.resize(fitted_mask, target_fitted_bf.shape[::-1],
                               interpolation=cv2.INTER_LANCZOS4)
        save_path = masks[dname].parent.joinpath("Norm_Mask.tif")
        ski.io.imsave(save_path, norm_mask)
        console.print(f"{dname} : '{save_path}'")
        console.line()
    
    console.line()
    console.print("[green]Done! \n")
    # -------------------------------------------------------------------------/