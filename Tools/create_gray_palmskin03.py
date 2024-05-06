import sys
from pathlib import Path

import cv2
from rich.console import Console

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.processeddatainstance import ProcessedDataInstance
# -----------------------------------------------------------------------------/

console = Console()
processed_di = ProcessedDataInstance()
processed_di.parse_config({"data_processed": {"instance_desc": "20240219_fixmm3d"}})
processed_di.palmskin_processed_dname_dirs_dict

with console.status("running..."):
    for path in processed_di.palmskin_processed_dname_dirs_dict.values():
        
        # source
        source = path.joinpath("03_RGB_direct_max_zproj.tif")
        img = cv2.imread(str(source))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # target
        target = path.joinpath("03_RGB_direct_max_zproj2Gray.tif")
        if not target.exists():
            cv2.imwrite(str(target), img)