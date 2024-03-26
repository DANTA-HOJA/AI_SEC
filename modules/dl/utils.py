from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, r2_score

from ..assert_fn import *
from ..shared.clioutput import CLIOutput
# -----------------------------------------------------------------------------/


def set_gpu(cuda_idx:int, cli_out:CLIOutput=None):
    """
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Can't find any GPU")
    
    device = torch.device(f"cuda:{cuda_idx}")
    
    # torch.cuda.set_device(cuda_idx)
    # device_num = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    torch.cuda.empty_cache()
    
    if cli_out: cli_out.write(f"Using '{device}', device_name = '{device_name}'")
    
    return device
    # -------------------------------------------------------------------------/



def gen_class2num_dict(num2class_list:List[str]):
    """
    """
    return {cls:i for i, cls in enumerate(num2class_list)}
    # -------------------------------------------------------------------------/



def gen_class_counts_dict(dataset_df:pd.DataFrame, num2class_list:List[str]):
    """
    """
    counter = Counter(dataset_df["class"])
        
    class_counts_dict: Dict[str, int] = {}
    for cls in num2class_list:
        class_counts_dict[cls] = counter[cls]
    
    return class_counts_dict
    # -------------------------------------------------------------------------/



def get_fish_path(image_name:str, df_dataset_xlsx:pd.DataFrame):
    """
    """
    df_filtered_rows = df_dataset_xlsx[(df_dataset_xlsx['image_name'] == image_name)]
    fish_path = list(df_filtered_rows["path"])[0]
    
    return Path(fish_path)
    # -------------------------------------------------------------------------/



def get_fish_class(image_name:str, df_dataset_xlsx:pd.DataFrame):
    """
    """
    df_filtered_rows = df_dataset_xlsx[(df_dataset_xlsx['image_name'] == image_name)]
    fish_class = list(df_filtered_rows["class"])[0]
    
    return fish_class
    # -------------------------------------------------------------------------/



def calculate_metrics(log:Dict, average_loss:float,
                      predict_list:list, groundtruth_list:list,
                      class2num_dict:Dict[str, int]):
    """
    """
    """ Calculate different f1-score """
    class_f1 = f1_score(groundtruth_list, predict_list, average=None) # by class
    micro_f1 = f1_score(groundtruth_list, predict_list, average='micro')
    macro_f1 = f1_score(groundtruth_list, predict_list, average='macro')
    weighted_f1 = f1_score(groundtruth_list, predict_list, average='weighted')
    maweavg_f1 = (macro_f1 + weighted_f1)/2
    
    """ Update `average_loss` """
    if average_loss is not None: log["average_loss"] = round(average_loss, 5)
    else: log["average_loss"] = None
    
    # deal with `class_f1` ( 很可能會缺其中一種 label, 例如: 'BG' )
    exist_labels = np.unique(predict_list + groundtruth_list) # np.unique 自帶 sorting
    if np.issubdtype(exist_labels.dtype, np.unicode_): # ndarray 是否為 Unicode string
        tmp_dict = {class2num_dict[label]: f1 for label, f1 in zip(exist_labels, class_f1)}
    else:
        tmp_dict = {label: f1 for label, f1 in zip(exist_labels, class_f1)}
    for key, value in class2num_dict.items():
        try:
            log[f"{key}_f1"] = round(tmp_dict[value], 5)
        except KeyError:
            log[f"{key}_f1"] = "---"
    
    """ Update other `f1-score` """
    log["micro_f1"] = round(micro_f1, 5)
    log["macro_f1"] = round(macro_f1, 5)
    log["weighted_f1"] = round(weighted_f1, 5)
    log["maweavg_f1"] = round(maweavg_f1, 5)
    # -------------------------------------------------------------------------/


def calculate_r_squared(log:Dict, average_loss:float,
                        predict_list:list, groundtruth_list:list):
    """
    """
    """ Update `average_loss` """
    if average_loss is not None: log["average_loss"] = round(average_loss, 5)
    else: log["average_loss"] = None
    
    score = r2_score(groundtruth_list, predict_list)
    log["r_squared"] = round(score, 5)
    # -------------------------------------------------------------------------/