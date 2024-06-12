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
        self.logger_name = None
        
        self._set_logger(logger_name)
        # ---------------------------------------------------------------------/



    def _set_logger(self, logger_name:str):
        """ 
        """
        if logger_name == "":
            self.logger_name = None
            self._logger = None
        else:
            self.logger_name = logger_name
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



    def divide(self, title:str=None,
                    length:int=100, characters:str="="):
        """
        """
        if self._display_on_CLI:
            
            div_length = length
            char = characters
            text = f"{Fore.GREEN}{char*div_length}{Style.RESET_ALL}"
            
            if title is not None:
                
                # trucate title
                if len(title) >= (div_length-10):
                    title = f"{title[:46]} ..."
                
                # calculate spacing
                center = f" {title} "
                rest = div_length - len(center)
                left = int(rest/2)
                right = rest - left
                text = (f"{Fore.GREEN}{char*left}"
                        f"{Fore.YELLOW}{center}"
                        f"{Fore.GREEN}{char*right}{Style.RESET_ALL}")
            
            tqdm.write(f"\n{text}\n")
        # ---------------------------------------------------------------------/



    def new_line(self):
        """
        """
        if self._display_on_CLI:
            tqdm.write("") # empty new line
        # ---------------------------------------------------------------------/