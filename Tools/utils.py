import os
from pathlib import Path
# -----------------------------------------------------------------------------/


def get_tool_config(path:str) -> Path:
    """
    """
    path = os.path.splitext(path)[0]
    up_dir, name = os.path.split(path)
    config_path = Path(up_dir).joinpath("config", f"{name}.toml")
    
    return config_path
    # -------------------------------------------------------------------------/
    