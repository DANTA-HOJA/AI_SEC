from pathlib import Path
from typing import List, Dict, Tuple, Union

import pandas as pd
import torch
from sklearn.metrics import f1_score

from ..shared.clioutput import CLIOutput
# -----------------------------------------------------------------------------/


def set_gpu(cuda_idx:int, cli_out:CLIOutput=None):
    """
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Can't find any GPU")
    
    device = torch.device("cuda")
    
    torch.cuda.set_device(cuda_idx)
    device_num = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_num)
    torch.cuda.empty_cache()
    
    if cli_out: cli_out.write(f"Using '{device}', device_name = '{device_name}'")
    
    return device
    # -------------------------------------------------------------------------/



def gen_class2num_dict(num2class_list:List[str]):
    """
    """
    return {cls:i for i, cls in enumerate(num2class_list)}
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



def calculate_class_weight(class_count_dict:Dict[str, int]) -> torch.Tensor:
    """ To calculate `class_weight` for Loss function in `List` format

        - Calculate `class_weight`: https://naadispeaks.wordpress.com/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/
        - Apply `class_weight`: https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731 
    
    Args:
        class_count_dict (Dict[str, int]): A `dict` contains the statistic information for each class, 
        e.g. `{ "L": 450, "M": 740, "S": 800 }`

    Returns:
        torch.tensor: `class_weight` in `torch.Tensor` format
    """
    class_weights_list = []
    total_samples = sum(class_count_dict.values())
    
    for key, value in class_count_dict.items(): # value = number of samples of the class
        class_weights_list.append((1 - (value/total_samples)))

    return torch.tensor(class_weights_list, dtype=torch.float)
    # -------------------------------------------------------------------------/



def calculate_metrics(log:Dict, average_loss:float,
                      predict_list:List[int], groundtruth_list:List[int],
                      class2num_dict:Dict[str, int]):
    
    """ Calculate different f1-score """
    class_f1 = f1_score(groundtruth_list, predict_list, average=None) # by class
    micro_f1 = f1_score(groundtruth_list, predict_list, average='micro')
    macro_f1 = f1_score(groundtruth_list, predict_list, average='macro')
    weighted_f1 = f1_score(groundtruth_list, predict_list, average='weighted')
    maweavg_f1 = (macro_f1 + weighted_f1)/2
    
    """ Update `average_loss` """
    if average_loss is not None: log["average_loss"] = round(average_loss, 5)
    else: log["average_loss"] = None
    
    """ Update `f1-score` """
    for key, value in class2num_dict.items(): log[f"{key}_f1"] = round(class_f1[value], 5)
    log["micro_f1"] = round(micro_f1, 5)
    log["macro_f1"] = round(macro_f1, 5)
    log["weighted_f1"] = round(weighted_f1, 5)
    log["maweavg_f1"] = round(maweavg_f1, 5)
    # -------------------------------------------------------------------------/