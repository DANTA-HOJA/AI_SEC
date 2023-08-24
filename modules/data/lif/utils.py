import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from logging import Logger
import json

from ...shared.utils import decide_cli_output
from ...assert_fn import *


def scan_lifs_under_dir(dir:Path, batches:list, logger:Logger=None) -> List[str]:
    """ Scan Leica LIF file
    """
    cli_out = decide_cli_output(logger)
    
    assert_dir_exists(dir)
    lif_path_list = []
    
    if batches:
        for batch in batches:
            found_list = list(dir.joinpath(batch).glob("**/*.lif"))
            lif_path_list.extend(found_list)
    else:
        lif_path_list = list(dir.glob("**/*.lif"))
    
    lif_path_list = [str(lif_path) for lif_path in lif_path_list]
    lif_path_list.sort(key=lambda x: x.split(os.sep)[-1])
    
    """ CLI output """
    formatted = json.dumps(lif_path_list, indent=4)
    cli_out(f'lif_path_list {type(lif_path_list)}: {formatted}')
    cli_out(f"[ found {len(lif_path_list)} lif files ]")
    
    return lif_path_list