import os
from pathlib import Path
from typing import List, Dict, Tuple, Union
import json

import pandas as pd
import torch
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------/


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



def plot_training_trend(save_dir:Path, loss_key:str, score_key:str,
                        train_logs:List[dict], valid_logs:List[dict]):
    """
        figsize (resolution) = w*dpi, h*dpi
        
        e.g. (1200,600) pixels can be
            - figsize=(15,7.5), dpi= 80
            - figsize=(12,6)  , dpi=100
            - figsize=( 8,4)  , dpi=150
            - figsize=( 6,3)  , dpi=200 etc.
    """
    train_logs = pd.DataFrame(train_logs)
    valid_logs = pd.DataFrame(valid_logs)
    
    """ Create figure set """
    fig, axs = plt.subplots(1, 2, figsize=(14,6), dpi=100)
    fig.suptitle('Training')
    
    """ Loss figure """
    axs[0].plot(list(train_logs["epoch"]), list(train_logs[loss_key]), label="train")
    axs[0].plot(list(valid_logs["epoch"]), list(valid_logs[loss_key]), label="validate")
    axs[0].legend() # turn legend on
    axs[0].set_title(loss_key)

    """ Score figure """
    axs[1].plot(list(train_logs["epoch"]), list(train_logs[score_key]), label="train")
    axs[1].plot(list(valid_logs["epoch"]), list(valid_logs[score_key]), label="validate")
    axs[1].set_ylim(0.0, 1.1)
    axs[1].legend() # turn legend on
    axs[1].set_title(score_key)

    """ Save figure """
    fig_path = save_dir.joinpath(f"training_trend_{score_key}.png")
    fig.savefig(fig_path)
    plt.close(fig) # Close figure
    # -------------------------------------------------------------------------/



def save_training_logs(save_dir:Path, train_logs:List[dict],
                       valid_logs:List[dict], best_val_log:dict):
    """
    """
    """ train_logs """
    df_train_logs = pd.DataFrame(train_logs)
    df_train_logs.set_index("epoch", inplace=True)

    """ valid_logs """
    df_valid_logs = pd.DataFrame(valid_logs)
    df_valid_logs.set_index("epoch", inplace=True)
    
    """ Concat two logs """
    concat_df = pd.concat([df_train_logs, df_valid_logs], axis=1)
    
    """ Save log """
    path = save_dir.joinpath(r"{Logs}_training_log.xlsx")
    concat_df.to_excel(path, engine="openpyxl")
    
    """ best_val_log """
    path = save_dir.joinpath(r"{Logs}_best_valid.log")
    with open(path, mode="w") as f_writer:
        json.dump(best_val_log, f_writer, indent=4)
    # -------------------------------------------------------------------------/



def save_model(desc:str, save_dir:Path, model_state_dict:dict, optimizer_state_dict:dict):
    """
    WARNING:
    
    - If you only plan to keep the best performing model (according to the acquired validation loss),
        don't forget that `best_model_state = model.state_dict()` returns a reference to the state and not its copy!
    
    - You must serialize best_model_state or use `best_model_state = deepcopy(model.state_dict())`,
        otherwise your best best_model_state will keep getting updated by the subsequent training iterations.
        As a result, the final model state will be the state of the overfitted model.
    
    - ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    
    """
    """ Composite dicts """
    temp_dict: dict = {}
    temp_dict["model_state_dict"] = model_state_dict
    temp_dict["optimizer_state_dict"] = optimizer_state_dict
    
    torch.save(temp_dict, save_dir.joinpath(f"{desc}_model.pth"))
    # -------------------------------------------------------------------------/



def rename_training_dir(orig_dir:Path, time_stamp:str, state:str,
                        epochs:int, aug_on_fly:bool, use_hsv:bool):
    """
    """
    temp_str = f"{epochs}_epochs"
    if aug_on_fly: temp_str += "_AugOnFly"
    if use_hsv: temp_str += "_HSV"
    new_name = f"{time_stamp}_{{{state}}}_{{{temp_str}}}"
    
    orig_dir_split = str(orig_dir).split(os.sep)
    orig_dir_split[-1] = new_name
    new_dir = os.sep.join(orig_dir_split)
    
    os.rename(orig_dir, new_dir)
    # -------------------------------------------------------------------------/