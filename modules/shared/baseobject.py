import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

from rich.progress import *
from tomlkit.toml_document import TOMLDocument

from .clioutput import CLIOutput
from .config import load_config
from .pathnavigator import PathNavigator
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