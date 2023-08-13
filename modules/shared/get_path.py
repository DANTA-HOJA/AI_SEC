import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from logging import Logger
from tomlkit.toml_document import TOMLDocument

from .utils import decide_cli_output, load_config

from ..assert_fn import *
from ..assert_fn import assert_0_or_1_instance_root, assert_0_or_1_processed_dir


__all__ = ["PathNavigator"]



class PathNavigator():
    
    def __init__(self) -> None:
        """
        """
        self.dbpp = _DBPPNavigator()
        self.raw_data = _RAWDataPathNavigator()
        self.processed_data = _ProcessedDataPath()



class _DBPPNavigator():
    
    def __init__(self) -> None:
        """
        """
        """ Load `dbpp_config` """
        self.dbpp_config = load_config("db_path_plan.toml")
    
    
    def get_fiji_local_dir(self, display_on_CLI:bool=False, logger:Logger=None) -> str:
        """
        """
        cli_out = decide_cli_output(logger)
        
        """ `dbpp_config` keywords """
        fiji_local = Path(self.dbpp_config["fiji_local"])
        assert_dir_exists(fiji_local)
        
        """ CLI output """
        if display_on_CLI:
            cli_out(f"Fiji Local: '{fiji_local}'")
        
        return str(fiji_local)
    
    
    def get_one_of_dbpp_roots(self, dbpp_key:str, display_on_CLI:bool=False, logger:Logger=None) -> Path:
        """
        """
        cli_out = decide_cli_output(logger)
        
        """ `dbpp_config` keywords """
        db_root = Path(self.dbpp_config["root"])
        assert_dir_exists(db_root)
        chosen_root = db_root.joinpath(self.dbpp_config[dbpp_key])
        assert_dir_exists(chosen_root)
        
        """ CLI output """
        if display_on_CLI:
            str_split = dbpp_key.split("_")
            abbr_list = ["nasdl", "cmd"]
            for word in str_split:
                if word in abbr_list: abbr_list.append(word.upper())
                else: word = abbr_list.append(word.capitalize())
            cli_out(f"{' '.join(abbr_list[2:])} Root: '{chosen_root}'")
            
        return chosen_root



class _RAWDataPathNavigator():
    
    def __init__(self) -> None:
        """
        """
        self.dbpp = _DBPPNavigator()
    

    def get_lif_scan_root(self, config:Union[dict, TOMLDocument],
                          display_on_CLI:bool=False, logger:Logger=None) -> Path:
        """
        """
        cli_out = decide_cli_output(logger)
        
        """ config keywords """
        nasdl_dir = config["data_nasdl"]["dir"]
        nasdl_type = config["data_nasdl"]["type"]
        
        """ Generate path """
        data_nasdl_root = self.dbpp.get_one_of_dbpp_roots("data_nasdl")
        lif_scan_root = data_nasdl_root.joinpath(nasdl_dir, nasdl_type)
        assert_dir_exists(lif_scan_root)
        
        """ CLI output """
        if display_on_CLI:
            cli_out(f"LIF Scan Root: '{lif_scan_root}'")
        
        return lif_scan_root



class _ProcessedDataPath():
    
    def __init__(self) -> None:
        """
        """
        self.dbpp = _DBPPNavigator()
    
    
    def get_instance_root(self, config:Union[dict, TOMLDocument],
                          display_on_CLI:bool=False, logger:Logger=None) -> Path:
        """
        """
        cli_out = decide_cli_output(logger)
        
        """ config keywords """
        instance_desc = config["data_processed"]["instance_desc"]
        
        """ Scan path """
        data_processed_root = self.dbpp.get_one_of_dbpp_roots("data_processed")
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
    
    
    def get_processed_dir(self, image_type:str, config:Union[dict, TOMLDocument],
                          display_on_CLI:bool=False, logger:Logger=None) -> Path:
        """ Get one of processed directories,
        
        1. `{[palmskin_reminder]}_PalmSkin_preprocess` or
        2. `{[brightfield_reminder]}_BrightField_analyze`
        
        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        cli_out = decide_cli_output(logger)
        
        """ Assign `target_text` """
        if image_type == "palmskin":
            target_text = "PalmSkin_preprocess"
        elif image_type == "brightfield":
            target_text = "BrightField_analyze"
        else: raise ValueError(f"Can't recognize arg: '{image_type}'")
        
        """ Scan path """
        instance_root = self.get_instance_root(config)
        found_list = list(instance_root.glob(f"*{target_text}*"))
        assert_0_or_1_processed_dir(found_list, target_text)
        
        """ Assign path """
        if found_list:
            processed_dir = found_list[0]
        else:
            reminder = config["data_processed"][f"{image_type}_reminder"]
            processed_dir = instance_root.joinpath(f"{{{reminder}}}_{target_text}")
        
        """ CLI output """
        if display_on_CLI:
            cli_out(f"{image_type.capitalize()} Processed Dir: '{processed_dir}'")
        
        return processed_dir