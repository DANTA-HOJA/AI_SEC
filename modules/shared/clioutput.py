from typing import List, Dict, Tuple, Union
from tqdm.auto import tqdm

from .logger import init_logger



class CLIOutput():
    
    def __init__(self, display_on_CLI:bool=True, logger_name:str="") -> None:
        """ 
        """
        self._display_on_CLI = display_on_CLI
        self._logger = None
        
        self.set_logger(logger_name)
    
    
    
    def set_logger(self, logger_name:str):
        """ 
        """
        if logger_name == "":
            self._logger = None
        else:
            self._logger = init_logger(logger_name)
    
    
    
    def write(self, message:str):
        """ 
        """
        if self._display_on_CLI:
            if self._logger:
                self._logger.info(message)
            else:
                tqdm.write(message)