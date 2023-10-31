import os
import sys
from colorama import Fore, Style
from pathlib import Path

sys.path.append("./../../modules/") # add path to scan customized module
from logger import init_logger
from gallery.cam.CamGalleryCreator import CamGalleryCreator

# -------------------------------------------------------------------------------------

log = init_logger(r"Make Cam Gallery")
config_path = Path(r"./../../Config/mk_cam_gallery.toml")

# -------------------------------------------------------------------------------------

print(); log.info(f"Start {Fore.BLUE}`{os.path.basename(__file__)}`{Style.RESET_ALL}")
log.info(f"config_path: {Fore.GREEN}'{config_path.resolve()}'{Style.RESET_ALL}\n")

cam_gallery_creator = CamGalleryCreator(config_path=config_path)
cam_gallery_creator.run()

print(); log.info(f"Done {Fore.BLUE}`{os.path.basename(__file__)}`{Style.RESET_ALL}\n")