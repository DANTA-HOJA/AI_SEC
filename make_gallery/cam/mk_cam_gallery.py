import os
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

sys.path.append("./../../modules/") # add path to scan customized module
from gallery.cam.CamGalleryCreator import CamGalleryCreator



config_path = Path(r"/home/rime97410000/ZebraFish_Code/ZebraFish_AP_POS/make_gallery/cam/mk_cam_gallery.toml")

cam_gallery_creator = CamGalleryCreator(config_path=config_path)
cam_gallery_creator.run()