import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from logging import Logger
from tomlkit.toml_document import TOMLDocument

from ..misc.utils import decide_cli_output, load_config

from ..assert_fn import *


def get_fiji_local_path(logger:Logger=None) -> Path:
    """
    """
    cli_out = decide_cli_output(logger)
    
    """ read `dbpp_config` """
    dbpp_config = load_config("db_path_plan.toml")
    
    """ `dbpp_config` keywords """
    fiji_local = Path(dbpp_config["fiji_local"])
    assert_dir_exists(fiji_local)
    
    """ CLI output """
    cli_out(f"Fiji Local: '{fiji_local}'")
    
    return str(fiji_local)



def get_lif_scan_root(config:Union[dict, TOMLDocument], logger:Logger=None):
    """
    """
    cli_out = decide_cli_output(logger)
    
    """ read `dbpp_config` """
    dbpp_config = load_config("db_path_plan.toml")
    
    """ `dbpp_config` keywords """
    db_root = Path(dbpp_config["root"])
    assert_dir_exists(db_root)
    data_nasdl = dbpp_config["data_nasdl"]
    
    """ config keywords """
    nasdl_dir = config["data_nasdl"]["dir"]
    nasdl_type = config["data_nasdl"]["type"]
    
    """ Generate path """
    lif_scan_root = db_root.joinpath(data_nasdl, nasdl_dir, nasdl_type)
    assert_dir_exists(db_root)
    
    """ CLI output """
    cli_out(f"LIF Scan Root: '{lif_scan_root}'")
    
    return lif_scan_root