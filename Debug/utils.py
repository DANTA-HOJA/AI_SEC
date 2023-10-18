import os
from pathlib import Path
# -----------------------------------------------------------------------------/


def get_debug_config(__file__:str) -> Path:
    """
    """
    path = os.path.splitext(__file__)[0]
    up_dir, name = os.path.split(path)
    config_path = Path(up_dir).joinpath("Config", f"{name}.toml")
    
    return config_path
    # -------------------------------------------------------------------------/
    