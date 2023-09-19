import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

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