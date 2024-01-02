import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

from tomlkit.toml_document import TOMLDocument

from ..assert_fn import *
from ..assert_fn import (assert_0_or_1_instance_root,
                         assert_0_or_1_processed_dir,
                         assert_0_or_1_recollect_dir)
from .clioutput import CLIOutput
from .config import load_config

__all__ = ["PathNavigator"]
# -----------------------------------------------------------------------------/


class PathNavigator:

    def __init__(self) -> None:
        """
        """
        self.dbpp = _DBPPNavigator()
        self.raw_data = _RAWDataPathNavigator()
        self.processed_data = _ProcessedDataPath()
        # ---------------------------------------------------------------------/



class _DBPPNavigator:

    def __init__(self) -> None:
        """
        """
        """ Load `dbpp_config` """
        self.dbpp_config = load_config("db_path_plan.toml")
        # ---------------------------------------------------------------------/


    def get_fiji_local_dir(self, cli_out:CLIOutput=None) -> Path:
        """
        """
        """ `dbpp_config` keywords """
        fiji_local = Path(self.dbpp_config["fiji_local"])
        assert_dir_exists(fiji_local)
        
        """ CLI output """
        if cli_out: cli_out.write(f"Fiji Local: '{fiji_local}'")
        
        return fiji_local
        # ---------------------------------------------------------------------/


    def get_one_of_dbpp_roots(self, dbpp_key:str, cli_out:CLIOutput=None) -> Path:
        """
        """
        """ `dbpp_config` keywords """
        db_root = Path(self.dbpp_config["root"])
        assert_dir_exists(db_root)
        chosen_root = db_root.joinpath(self.dbpp_config[dbpp_key])
        assert_dir_exists(chosen_root)
        
        """ CLI output """
        if cli_out:
            str_split = dbpp_key.split("_")
            abbr_list = ["nasdl", "cmd"]
            for word in str_split:
                if word in abbr_list: abbr_list.append(word.upper())
                else: word = abbr_list.append(word.capitalize())
            cli_out.write(f"{' '.join(abbr_list[2:])} Root: '{chosen_root}'")
            
        return chosen_root
        # ---------------------------------------------------------------------/



class _RAWDataPathNavigator:

    def __init__(self) -> None:
        """
        """
        self.dbpp = _DBPPNavigator()
        # ---------------------------------------------------------------------/


    def get_lif_scan_root(self, config:Union[dict, TOMLDocument],
                          cli_out:CLIOutput=None) -> Path:
        """
        """
        """ config keywords """
        nasdl_dir = config["data_nasdl"]["dir"]
        nasdl_type = config["data_nasdl"]["type"]
        
        """ Generate path """
        data_nasdl_root = self.dbpp.get_one_of_dbpp_roots("data_nasdl")
        lif_scan_root = data_nasdl_root.joinpath(nasdl_dir, nasdl_type)
        assert_dir_exists(lif_scan_root)
        
        """ CLI output """
        if cli_out: cli_out.write(f"LIF Scan Root: '{lif_scan_root}'")
        
        return lif_scan_root
        # ---------------------------------------------------------------------/



class _ProcessedDataPath:

    def __init__(self) -> None:
        """
        """
        self.dbpp = _DBPPNavigator()
        # ---------------------------------------------------------------------/


    def get_instance_root(self, config:Union[dict, TOMLDocument],
                          cli_out:CLIOutput=None) -> Path:
        """
        """
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
        if cli_out: cli_out.write(f"Instance Root: '{instance_root}'")
        
        return instance_root
        # ---------------------------------------------------------------------/


    def get_processed_dir(self, image_type:str, config:Union[dict, TOMLDocument],
                          cli_out:CLIOutput=None):
        """ Get one of processed directories,
        
        1. `{[palmskin_reminder]}_PalmSkin_preprocess` or
        2. `{[brightfield_reminder]}_BrightField_analyze`
        
        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        """ Assign `target_text` """
        if image_type == "palmskin":
            target_text = "PalmSkin_preprocess"
        elif image_type == "brightfield":
            target_text = "BrightField_analyze"
        else: raise ValueError(f"Can't recognize arg: '{image_type}'")
        
        """ Scan path """
        instance_root = self.get_instance_root(config)
        found_list = list(instance_root.glob(f"{{*}}_{target_text}"))
        assert_0_or_1_processed_dir(found_list, target_text)
        
        """ Assign path """
        if found_list:
            processed_dir = found_list[0]
            """ CLI output """
            if cli_out: cli_out.write(f"{image_type.capitalize()} Processed Dir: '{processed_dir}'")
        else:
            processed_dir = None
        
        return processed_dir
        # ---------------------------------------------------------------------/


    def get_recollect_dir(self, image_type:str, config:Union[dict, TOMLDocument],
                          cli_out:CLIOutput=None):
        """ Get one of recollect directories,
        
        1. `{[palmskin_reminder]}_PalmSkin_reCollection` or
        2. `{[brightfield_reminder]}_BrightField_reCollection`
        
        Args:
            image_type (str): `palmskin` or `brightfield`
        """
        """ Assign `target_text` """
        if image_type == "palmskin":
            target_text = "PalmSkin_reCollection"
        elif image_type == "brightfield":
            target_text = "BrightField_reCollection"
        else: raise ValueError(f"Can't recognize arg: '{image_type}'")
        
        """ Scan path """
        instance_root = self.get_instance_root(config)
        found_list = list(instance_root.glob(f"{{*}}_{target_text}"))
        assert_0_or_1_recollect_dir(found_list, target_text)
        
        """ Assign path """
        if found_list:
            recollect_dir = found_list[0]
            """ CLI output """
            if cli_out: cli_out.write(f"{image_type.capitalize()} Recollect Dir: '{recollect_dir}'")
        else:
            recollect_dir = None
        
        return recollect_dir
        # ---------------------------------------------------------------------/


    def get_data_xlsx_path(self, config:Union[dict, TOMLDocument], cli_out:CLIOutput=None):
        """
        """
        instance_root = self.get_instance_root(config)
        data_xlsx_path = instance_root.joinpath("data.xlsx")
        
        if data_xlsx_path.exists():
            """ CLI output """
            if cli_out: cli_out.write(f"data.xlsx : '{data_xlsx_path}'")
        else:
            data_xlsx_path = None
        
        return data_xlsx_path
        # ---------------------------------------------------------------------/


    def get_clustered_xlsx_dir(self, config:Union[dict, TOMLDocument], cli_out:CLIOutput=None):
        """
        """
        instance_root = self.get_instance_root(config)
        clustered_xlsx_dir = instance_root.joinpath("Clustered_xlsx")
        
        if clustered_xlsx_dir.exists():
            """ CLI output """
            if cli_out: cli_out.write(f"Clustered XLSX Dir: '{clustered_xlsx_dir}'")
        else:
            clustered_xlsx_dir = None
        
        return clustered_xlsx_dir
        # ---------------------------------------------------------------------/