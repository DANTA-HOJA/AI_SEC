import sys
from pathlib import Path

import cv2
from rich import print
from rich.console import Console
from rich.pretty import Pretty
from rich.traceback import install

pkg_dir = Path(__file__).parents[1] # `dir_depth` to `repo_root`
if (pkg_dir.exists()) and (str(pkg_dir) not in sys.path):
    sys.path.insert(0, str(pkg_dir)) # add path to scan customized package

from modules.data.processeddatainstance import ProcessedDataInstance
from modules.shared.clioutput import CLIOutput
from modules.shared.utils import get_repo_root

install()
# -----------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")

    """ Init components """
    console = Console()
    cli_out = CLIOutput()
    processed_di = ProcessedDataInstance()

    processed_di.parse_config("0.2.3.process_palmskin_manualroi.toml")
    print("", Pretty(processed_di.config, expand_all=True))
    cli_out.divide()

    with console.status("running..."):
        for path in processed_di.palmskin_processed_dname_dirs_dict.values():
            
            # target (dst path)
            target = path.joinpath("03_RGB_direct_max_zproj2Gray.tif")
            if not target.exists():
                
                # read source image (03 color)
                source = path.joinpath("03_RGB_direct_max_zproj.tif")
                img = cv2.imread(str(source))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                cv2.imwrite(str(target), img)
                print(f"[ {target.parent.stem} : '{target}' ]")