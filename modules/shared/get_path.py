import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from logging import Logger
from tomlkit.toml_document import TOMLDocument

from utils import decide_cli_output, get_target_str_idx_in_list

from ..assert_fn import *
from ..assert_fn import assert_run_under_repo_root, assert_0_or_1_instance_root, \
                        assert_0_or_1_palmskin_preprocess_dir


def get_repo_root() -> Path:
    """ TODO
    """
    
    path_split = os.path.abspath(".").split(os.sep)
    target_idx = get_target_str_idx_in_list(path_split, "ZebraFish_AP_POS")
    assert_run_under_repo_root(target_idx)
    
    """ generate path """
    repo_root = os.sep.join(path_split[:target_idx+1])
    
    return Path(repo_root)



def get_fiji_local_path(dbpp_config:Union[dict, TOMLDocument], logger:Logger=None) -> str:
    """
    """
    cli_out = decide_cli_output(logger)
    
    """ `dbpp_config` keywords """
    fiji_local = Path(dbpp_config["fiji_local"])
    assert_dir_exists(fiji_local)
    
    """ CLI output """
    cli_out(f"Fiji Local: '{fiji_local}'")
    
    return str(fiji_local)



def get_lif_scan_root(dbpp_config:Union[dict, TOMLDocument],
                      config:Union[dict, TOMLDocument], logger:Logger=None) -> Path:
    """
    """
    cli_out = decide_cli_output(logger)
    
    """ `dbpp_config` keywords """
    db_root = Path(dbpp_config["root"])
    assert_dir_exists(db_root)
    data_nasdl = dbpp_config["data_nasdl"]
    
    """ config keywords """
    nasdl_dir = config["data_nasdl"]["dir"]
    nasdl_type = config["data_nasdl"]["type"]
    
    """ Generate path """
    lif_scan_root = db_root.joinpath(data_nasdl, nasdl_dir, nasdl_type)
    assert_dir_exists(lif_scan_root)
    
    """ CLI output """
    cli_out(f"LIF Scan Root: '{lif_scan_root}'")
    
    return lif_scan_root



def get_instance_root(dbpp_config:Union[dict, TOMLDocument],
                      config:Union[dict, TOMLDocument], logger:Logger=None) -> Path:
    """
    """
    cli_out = decide_cli_output(logger)
    
    """ `dbpp_config` keywords """
    db_root = Path(dbpp_config["root"])
    assert_dir_exists(db_root)
    
    """ config keywords """
    instance_desc = config["data_processed"]["instance_desc"]
    
    """ Scan path """
    data_processed_root = db_root.joinpath(dbpp_config["data_processed"])
    assert_dir_exists(data_processed_root)
    found_list = list(data_processed_root.glob(f"*{instance_desc}*"))
    assert_0_or_1_instance_root(found_list, instance_desc)
    
    """ Assign path """
    if found_list:
        instance_root = found_list[0]
    else:
        instance_root = data_processed_root.joinpath(f"{{{instance_desc}}}_Academia_Sinica_iTBA")
    
    """ CLI output """
    cli_out(f"Instance Root: '{instance_root}'")
    
    return instance_root



def get_palmskin_preprocess_dir(dbpp_config:Union[dict, TOMLDocument],
                                config:Union[dict, TOMLDocument], logger:Logger=None) -> Path:
    """
    """
    cli_out = decide_cli_output(logger)
    
    """ `dbpp_config` keywords """
    db_root = Path(dbpp_config["root"])
    assert_dir_exists(db_root)
    
    """ config keywords """
    palmskin_reminder = config["data_processed"]["palmskin_reminder"]
    
    """ Scan path """
    instance_root = get_instance_root(dbpp_config, config, logger)
    found_list = list(instance_root.glob(f"*PalmSkin_preprocess*"))
    assert_0_or_1_palmskin_preprocess_dir(found_list)
    
    """ Assign path """
    if found_list:
        palmskin_preprocess_dir = found_list[0]
    else:
        palmskin_preprocess_dir = instance_root.joinpath(f"{{{palmskin_reminder}}}_PalmSkin_preprocess")
    
    """ CLI output """
    cli_out(f"Palmskin Preprocess Dir: '{palmskin_preprocess_dir}'")
    
    return palmskin_preprocess_dir