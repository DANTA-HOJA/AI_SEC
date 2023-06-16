from colorama import Fore, Back, Style
from tqdm.auto import tqdm

class CLIDivider():
    
    def __init__(self) -> None:
        pass
    
    
    def process_start(self, use_tqdm:bool=False):
        
        if use_tqdm: cli_out = tqdm.write
        else: cli_out = print
        
        cli_out("")


    def process_completed(self, use_tqdm:bool=False):
        
        if use_tqdm: cli_out = tqdm.write
        else: cli_out = print
        
        cli_out(f"{Fore.GREEN}{Back.BLACK}"); cli_out("="*100)
        cli_out("process complete !"); cli_out(f"{Style.RESET_ALL}")