import sys
from pathlib import Path

abs_module_path = Path("./../../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.plot.cam_gallery.creator.camgallerycreator import CamGalleryCreator
from modules.shared.utils import get_repo_root

import matplotlib; matplotlib.use("agg")
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

cam_gallery_creator = CamGalleryCreator()
cam_gallery_creator.run("4.make_cam_gallery.toml")