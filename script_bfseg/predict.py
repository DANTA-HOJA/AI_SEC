import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from rich import print
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from torch.utils.data import DataLoader

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

import models
from utils import BFSegTestSet, get_exist_bf_dirs, set_gpu, set_reproducibility

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.pathnavigator import PathNavigator
from modules.shared.utils import get_repo_root

console = Console(record=True)

install()
# -----------------------------------------------------------------------------/



if __name__ == "__main__":
    
    print(f"Repository: '{get_repo_root()}'")

    """ Init components """
    path_navigator = PathNavigator()
    processed_di = ProcessedDataInstance()
    processed_di.parse_config("bf_seg.toml")
    
    batch_size: int = 8
    model_name: str = "res18unet"
    device = set_gpu(0, console)
    set_reproducibility(2022) # 可能要 remove
    
    # get paths
    # -> path example: ".../{Data}_Processed/{20230827_test}_Academia_Sinica_i505/{autothres_triangle}_BrightField_analyze"
    path = processed_di.brightfield_processed_dir
    found_list = get_exist_bf_dirs(path, "*/02_cropped_BF.tif")
    if found_list == []:
        raise ValueError("Can't find any directories. Make sure that `data_processed.instance_desc` exists.")
    print(f"Found {len(found_list)} directories")
    
    # dataset, dataloader
    test_set = BFSegTestSet(found_list)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    save_paths: list = []
    
    # model
    num_class = 1
    if model_name == "unet":
        model = models.UNet(num_class).to(device)
    elif model_name == "res18unet":
        model = models.ResNetUNet(num_class).to(device)
    else:
        raise NotImplementedError
    
    # load `model_state_dict`
    time_stamp = processed_di.config["model_prediction"]["time_stamp"]
    
    bfseg_model_root: Path = \
        path_navigator.dbpp.get_one_of_dbpp_roots("model_bfseg")
    history_dirs = list(bfseg_model_root.glob(f"**/{time_stamp}*"))
    if len(history_dirs) == 1:
        history_dir = history_dirs[0]
        print(f"BFSeg Model: '{history_dir}'")
    elif len(history_dirs) == 0:
        raise ValueError("No `BFSeg` model matches the provided config. "
                         f"Got `time_stamp`: {time_stamp}.")
    else:
        raise ValueError("Duplicate `BFSeg` models match the provided config. "
                         f"Got `time_stamp`: {time_stamp}.")
    
    pth_file = history_dir.joinpath(f"{model_name}_best.pth")
    pth_file = torch.load(pth_file, map_location=device) # unpack to device directly
    model.load_state_dict(pth_file)
    
    
    model.eval()   # Set model to evaluate mode
    with torch.no_grad():
        for paths, inputs in track(test_loader,
                                   description=f"[yellow]test:"):
            
            inputs = inputs.to(device)
            preds = model(inputs)
            preds = preds.cpu().detach().numpy()
            # preds = torch.sigmoid(preds).cpu().detach().numpy()
            
            for path, pred_seg in zip(paths, preds):
                # postprocessing
                pred_seg = np.squeeze(pred_seg)
                pred_seg = cv2.resize(pred_seg, (1950, 700))
                pred_seg = np.clip(pred_seg, 0, 1)
                
                # overlapping orig and pred
                orig_path = Path(path).joinpath("02_cropped_BF.tif")
                orig: cv2.Mat = cv2.imread(str(orig_path), 0)
                orig = orig / 255.0
                overlap = (orig + pred_seg)*0.5
                
                # save predict mask
                save_path = Path(path).joinpath("UNet_predict_mask.tif")
                pred_seg = np.uint8(pred_seg*255)
                cv2.imwrite(str(save_path), pred_seg)
                save_paths.append(save_path)
                print(f"'{str(save_path)}'")
                
                # save overlap image
                save_path = Path(path).joinpath("UNet_cropped_BF--MIX.tif")
                overlap = np.uint8(overlap*255)
                cv2.imwrite(str(save_path), overlap)
                save_paths.append(save_path)
                print(f"'{str(save_path)}'")
                
    
    # for path in save_paths:
    #     os.remove(path)