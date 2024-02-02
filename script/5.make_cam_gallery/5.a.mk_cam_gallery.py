import sys
from pathlib import Path

pkg_dir = Path(__file__).parents[2] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.plot.cam_gallery.creator.camgallerycreator import CamGalleryCreator
from modules.shared.utils import get_repo_root

import matplotlib; matplotlib.use("agg")
# -----------------------------------------------------------------------------/

""" Detect Repository """
print(f"Repository: '{get_repo_root()}'")

cam_gallery_creator = CamGalleryCreator()
cam_gallery_creator.run("5.make_cam_gallery.toml")