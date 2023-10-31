import os
from pathlib import Path
# -----------------------------------------------------------------------------/


def get_tool_config_path(file_path:str) -> Path:
    """
    """
    file_path = os.path.splitext(file_path)[0]
    file_dir, name = os.path.split(file_path)
    config_path = Path(file_dir).joinpath("config", f"{name}.toml")
    
    return config_path
    # -------------------------------------------------------------------------/
    