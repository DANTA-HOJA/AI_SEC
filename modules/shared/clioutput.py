from typing import List, Dict, Tuple, Union
from colorama import Fore, Back, Style
from tqdm.auto import tqdm

from .logger import init_logger
# -----------------------------------------------------------------------------/


class CLIOutput():


    def __init__(self, display_on_CLI:bool=True, logger_name:str="") -> None:
        """ 
        """
        self._display_on_CLI = display_on_CLI
        self._logger = None
        
        self._set_logger(logger_name)
        # ---------------------------------------------------------------------/



    def _set_logger(self, logger_name:str):
        """ 
        """
        if logger_name == "":
            self._logger = None
        else:
            self._logger = init_logger(logger_name)
        # ---------------------------------------------------------------------/



    def write(self, message:str):
        """ 
        """
        if self._display_on_CLI:
            if self._logger:
                self._logger.info(message)
            else:
                tqdm.write(message)
        # ---------------------------------------------------------------------/



    def divide(self):
        """
        """
        if self._display_on_CLI:
            tqdm.write(f"\n{Fore.GREEN}{'='*100}{Style.RESET_ALL}\n")
        # ---------------------------------------------------------------------/



    def new_line(self):
        """
        """
        if self._display_on_CLI:
            tqdm.write("\n")
        # ---------------------------------------------------------------------/