import shutil
import sys
from pathlib import Path

from flask_socketio import SocketIO
from rich.console import Console
from rich.pretty import Pretty
from rich.progress import Progress
from rich.traceback import install

from ..ml.calc_seg_feat import (count_area, count_average_size, count_element,
                                get_patch_sizes, update_ana_toml_file,
                                update_seg_analysis_dict)
from ..ml.seg_generate import single_cellpose_prediction
from ..ml.utils import get_cellpose_param_name
from ..shared.config import load_config
from ..shared.pathnavigator import PathNavigator
from ..shared.utils import get_repo_root

install()
# -----------------------------------------------------------------------------/

def cellpose_for_sec(img_paths: list[Path], config_name: str,
                     console: Console, socketio: SocketIO = None):
    """
    """
    """ Init components """
    path_navigator = PathNavigator()

    """ Load config """
    console.rule("Load config")
    config = load_config(config_name)
    # [FileSys]
    # src_dir: str       = config["Filesys"]["src_dir"]
    gen_suffixes: list = config["Filesys"]["gen_suffixes"]
    # [Cellpose]
    cp_model_name: str = config["Cellpose"]["cp_model_name"]
    channels: list     = config["Cellpose"]["channels"]
    merge: int         = config["Cellpose"]["merge"]
    debug_mode: bool   = config["Cellpose"]["debug_mode"]
    console.print("", Pretty(config, expand_all=True), "")

    """ Check model """
    cp_model_dir = path_navigator.dbpp.get_one_of_dbpp_roots("model_cellpose")
    cp_model_path = cp_model_dir.joinpath(cp_model_name)
    if cp_model_path.is_file():
        seg_param_name = get_cellpose_param_name(config) # {model_name}_CH{0}{1}_M{}
    else:
        raise FileNotFoundError(f"'{cp_model_path}' is not a file or does not exist")

    """ Load model """
    if "cellpose" in sys.executable: # check python environment
            from cellpose import models as cpmodels
            cp_model = cpmodels.CellposeModel(gpu=True, pretrained_model=str(cp_model_path))
    else:
        raise RuntimeError("Detect environment name not for Cellpose. "
                            "Please follow the setup instructions provided at "
                            "'https://github.com/MouseLand/cellpose' "
                            "to create an environment.")

    """ Apply SLIC on each image """
    console.rule()
    with Progress(console=console) as pbar:
        task = pbar.add_task("[cyan]Processing...", total=len(img_paths))

        for enum, path in enumerate(img_paths, 1):
            
            # update pbar
            if socketio is not None: # WebUI
                socketio.emit('processing_progress', {
                    'filename': path.name,
                    'now': enum,
                    'total': len(img_paths)
                })
            pbar.advance(task)
            
            # create dir for each image
            dst_dir = path.parent.joinpath(path.stem)
            if dst_dir.is_dir():
                # 檢查 11 種檔案是否存在，有缺才 run
                suffixes = set([path.suffixes[-2] for path in sorted(dst_dir.glob("*"))])
                if suffixes == set(gen_suffixes):
                    continue
            else:
                dst_dir.mkdir()

            # run cellpose
            cell_seg, patch_seg = single_cellpose_prediction(dst_dir, path,
                                                                channels, cp_model, merge,
                                                                debug_mode)

            # update
            analysis_dict = {}
            analysis_dict = update_seg_analysis_dict(analysis_dict, *count_area(cell_seg))
            analysis_dict = update_seg_analysis_dict(analysis_dict, *count_element(cell_seg, "cell"))
            analysis_dict = update_seg_analysis_dict(analysis_dict, *count_element(patch_seg, "patch"))
            analysis_dict = update_seg_analysis_dict(analysis_dict, *count_average_size(analysis_dict, "cell"))
            analysis_dict = update_seg_analysis_dict(analysis_dict, *count_average_size(analysis_dict, "patch"))
            analysis_dict = update_seg_analysis_dict(analysis_dict, *get_patch_sizes(patch_seg))
            console.line()

            # update info to toml file
            ana_toml_file = dst_dir.joinpath(f"{path.stem}.ana.toml")
            update_ana_toml_file(ana_toml_file, analysis_dict)

            # move file
            # shutil.move(path, dst_dir)

        pbar.remove_task(task)
    # -------------------------------------------------------------------------/


if __name__ == '__main__':

    print(f"Repository: '{get_repo_root()}'")

    """ Init components """
    console = Console(record=True)

    """ Load config """
    console.rule("Load config")
    config = load_config("cp_seg.toml")
    # [FileSys]
    src_dir: str = config["Filesys"]["src_dir"]

    """ Scan image files """
    console.rule("Scan image files")
    img_paths = sorted(Path(src_dir).glob("*.tif*"))
    console.print(f"Total files: {len(img_paths)}")
    console.print(f"{[path.name for path in img_paths]}")
    
    cellpose_for_sec(img_paths, "cp_seg.toml", console)

    console.line()
    console.print("[green]Done! \n")
    # -------------------------------------------------------------------------/