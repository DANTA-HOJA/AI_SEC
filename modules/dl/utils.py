from pathlib import Path
from typing import List, Dict, Tuple, Union

import pandas as pd
import torch

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
    
    df_filtered_rows = df_dataset_xlsx[(df_dataset_xlsx['image_name'] == image_name)]
    fish_path = list(df_filtered_rows["path"])[0]
    
    return Path(fish_path)
    # -------------------------------------------------------------------------/