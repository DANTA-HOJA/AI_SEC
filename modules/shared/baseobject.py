import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

from rich.progress import *
from rich.traceback import install
from tomlkit.toml_document import TOMLDocument

from .clioutput import CLIOutput
from .config import load_config
from .pathnavigator import PathNavigator

install()
# -----------------------------------------------------------------------------/


class BaseObject:

    def __init__(self, display_on_CLI=True) -> None:
        """
        """
        # ---------------------------------------------------------------------
        # """ components """
        
        self._path_navigator = PathNavigator()
        self._cli_out = CLIOutput(display_on_CLI)
        self._pbar: Progress
        
        # ---------------------------------------------------------------------
        # """ attributes """
        
        self.config: Union[dict, TOMLDocument]
        self.src_root: Path
        self.dst_root: Path
        
        # ---------------------------------------------------------------------
        # """ actions """
        # TODO
        # ---------------------------------------------------------------------/


    def run(self, config:Union[str, Path]):
        """ Pre-execute below functions
            >>> self._cli_out.divide()
            >>> self._set_attrs(config)
        """
        self._cli_out.divide()
        self._set_attrs(config)
        self._check_dl_dataset_file_name()
        self._check_dl_pos_filtering()
        # ---------------------------------------------------------------------/


    def _set_attrs(self, config:Union[str, Path]):
        """ Pre-execute below functions
            >>> self.config = load_config(config, cli_out=self._cli_out)
            >>> self._set_config_attrs()
        """
        self.config = load_config(config, cli_out=self._cli_out)
        self._set_config_attrs()
        # ---------------------------------------------------------------------/


    def _set_config_attrs(self):
        """
        """
        raise NotImplementedError("This is a preset function in `BaseObject`, "
                                  "you should create a child class and replace this funtion.")
        # ---------------------------------------------------------------------/


    def _set_src_root(self):
        """
        """
        raise NotImplementedError("This is a preset function in `BaseObject`, "
                                  "you should create a child class and replace this funtion.")
        # ---------------------------------------------------------------------/


    def _set_dst_root(self):
        """
        """
        raise NotImplementedError("This is a preset function in `BaseObject`, "
                                  "you should create a child class and replace this funtion.")
        # ---------------------------------------------------------------------/


    def _reset_pbar(self):
        """
        """
        self._pbar = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TextColumn("{task.completed} of {task.total}"),
            auto_refresh=False
        )
        # ---------------------------------------------------------------------/


    def _check_dl_dataset_file_name(self):
        """
        """
        # check if the object is a dl component
        try:
            getattr(self, "model_name")
        except AttributeError:
            return
        
        cls_name = type(self).__name__
        
        if "NoCrop" in cls_name:
            if self.dataset_file_name != "DS_SURF3C_NOCROP.csv":
                raise ValueError(f"The expected (config) `dataset.file_name` "
                                f"for `{cls_name}` is "
                                f"'DS_SURF3C_NOCROP.csv', "
                                f"but got '{self.dataset_file_name}'")
        elif "SurfDGT" in cls_name:
            if self.dataset_file_name != "DS_SURFDGT.csv":
                raise ValueError(f"The expected (config) `dataset.file_name` "
                                f"for `{cls_name}` is "
                                f"'DS_SURFDGT.csv', "
                                f"but got '{self.dataset_file_name}'")
        else:
            if not ("CRPS" in self.dataset_file_name):
                raise ValueError(f"The expected (config) `dataset.file_name` "
                                f"for `{cls_name}` is like "
                                f"'DS_SURF3C_CRPS256_SF14_INT30_DRP65.csv', "
                                f"but got '{self.dataset_file_name}'")
        # ---------------------------------------------------------------------/


    def _check_dl_pos_filtering(self):
        """
        """
        # check if the object is a dl component
        try:
            getattr(self, "model_name")
        except AttributeError:
            return
        
        cls_name = type(self).__name__
        
        # get config
        if "Trainer" in cls_name: config_var = "config"
        elif "Tester" in cls_name: config_var = "training_config"
        config = getattr(self, config_var)
        note: str = config["note"]
        note = re.sub(r"[- ]", "", note).lower()
        
        for filter in ["aonly", "ponly"]:
            cond1 = (filter in cls_name.lower())
            cond2 = (filter in note)
            assert cond1 == cond2, (
                f"Using '{filter[0]} only' object: {cond1}, "
                f"'{filter[0]} only' is annotated in (config) `note`: {cond2}. "
                f"Make sure the conditions are the same before running. "
                f"Possible annotation for (config) `note`: "
                f"'{filter[0]} only', '{filter[0]} only' (case insensitive)")
        # ---------------------------------------------------------------------/
