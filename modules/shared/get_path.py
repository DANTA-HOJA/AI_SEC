import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from logging import Logger
from tomlkit.toml_document import TOMLDocument

from .utils import decide_cli_output

from ..assert_fn import *
from ..assert_fn import assert_0_or_1_instance_root, assert_0_or_1_palmskin_preprocess_dir


def get_fiji_local_dir(dbpp_config:Union[dict, TOMLDocument],
                       display_on_CLI:bool=False, logger:Logger=None) -> str:
    """
    """
    cli_out = decide_cli_output(logger)
    
    """ `dbpp_config` keywords """
    fiji_local = Path(dbpp_config["fiji_local"])
    assert_dir_exists(fiji_local)
    
    """ CLI output """
    if display_on_CLI:
        cli_out(f"Fiji Local: '{fiji_local}'")
    
    return str(fiji_local)



def get_lif_scan_root(dbpp_config:Union[dict, TOMLDocument],
                      config:Union[dict, TOMLDocument],
                      display_on_CLI:bool=False, logger:Logger=None) -> Path:
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
    if display_on_CLI:
        cli_out(f"LIF Scan Root: '{lif_scan_root}'")
    
    return lif_scan_root



def get_data_processed_root(dbpp_config:Union[dict, TOMLDocument],
                            display_on_CLI:bool=False, logger:Logger=None) -> Path:
    """
    """
    cli_out = decide_cli_output(logger)
    
    """ `dbpp_config` keywords """
    db_root = Path(dbpp_config["root"])
    assert_dir_exists(db_root)
    data_processed_root = db_root.joinpath(dbpp_config["data_processed"])
    assert_dir_exists(data_processed_root)
    
    """ CLI output """
    if display_on_CLI:
        cli_out(f"Data Processed Root: '{data_processed_root}'")
        
    return data_processed_root



def get_instance_root(dbpp_config:Union[dict, TOMLDocument],
                      config:Union[dict, TOMLDocument],
                      display_on_CLI:bool=False, logger:Logger=None) -> Path:
    """
    """
    cli_out = decide_cli_output(logger)
    
    """ config keywords """
    instance_desc = config["data_processed"]["instance_desc"]
    
    """ Scan path """
    data_processed_root = get_data_processed_root(dbpp_config)
    found_list = list(data_processed_root.glob(f"{{{instance_desc}}}*"))
    assert_0_or_1_instance_root(found_list, instance_desc)
    
    """ Assign path """
    if found_list:
        instance_root = found_list[0]
    else:
        instance_root = data_processed_root.joinpath(f"{{{instance_desc}}}_Academia_Sinica_iTBA")
    
    """ CLI output """
    if display_on_CLI:
        cli_out(f"Instance Root: '{instance_root}'")
    
    return instance_root



def get_palmskin_preprocess_dir(dbpp_config:Union[dict, TOMLDocument],
                                config:Union[dict, TOMLDocument],
                                display_on_CLI:bool=False, logger:Logger=None) -> Path:
    """
    """
    cli_out = decide_cli_output(logger)
    
    """ config keywords """
    palmskin_reminder = config["data_processed"]["palmskin_reminder"]
    
    """ Scan path """
    instance_root = get_instance_root(dbpp_config, config)
    found_list = list(instance_root.glob(f"*PalmSkin_preprocess*"))
    assert_0_or_1_palmskin_preprocess_dir(found_list)
    
    """ Assign path """
    if found_list:
        palmskin_preprocess_dir = found_list[0]
    else:
        palmskin_preprocess_dir = instance_root.joinpath(f"{{{palmskin_reminder}}}_PalmSkin_preprocess")
    
    """ CLI output """
    if display_on_CLI:
        cli_out(f"Palmskin Preprocess Dir: '{palmskin_preprocess_dir}'")
    
    return palmskin_preprocess_dir