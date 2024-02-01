import os
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Union
import toml
import tomlkit
from tomlkit.toml_document import TOMLDocument

from .clioutput import CLIOutput
from .utils import get_repo_root

from ..assert_fn import assert_only_1_config
# -----------------------------------------------------------------------------/


def load_config(config:Union[str, Path], reserve_comment:bool=False,
                cli_out:CLIOutput=None) -> Union[dict, TOMLDocument]:
    """ Scan and load the specific config under repo root

    Args:
        config (Union[str, Path]): full file name, like `abc.toml`
        reserve_comment (bool, optional): Defaults to False.
        cli_out (CLIOutput, optional): a `CLIOutput` object. Defaults to None.

    Raises:
        NotImplementedError: If (argument) `config` not `str` or `Path` object.

    Returns:
        Union[dict, TOMLDocument]: a toml config
    """
    if isinstance(config, dict):
        return config
    
    if reserve_comment:
        load_fn = tomlkit.load
    else:
        load_fn = toml.load
    
    path = None
    if isinstance(config, Path):
        path = config
    elif isinstance(config, str):
        repo_root = get_repo_root()
        found_list = list(repo_root.glob(f"**/{config}"))
        assert_only_1_config(found_list, config)
        path = found_list[0]
    else:
        raise NotImplementedError("Argument `config_file` should be `str` or `Path` object.")
    
    """ CLI output """
    if cli_out: cli_out.write(f"Config Path: '{path}'")
    
    with open(path, mode="r") as f_reader:
        config = load_fn(f_reader)
    
    return config
    # -------------------------------------------------------------------------/



def dump_config(path:Path, config:Union[dict, TOMLDocument]):
    """
    """
    with open(path, mode="w") as f_writer:
        tomlkit.dump(config, f_writer)
    # -------------------------------------------------------------------------/



def get_batch_config(file_path:str) -> List[Path]:
    """
    """
    file_path = os.path.splitext(file_path)[0]
    file_dir, name = os.path.split(file_path)
    config_dir = Path(file_dir).joinpath("batch_config")
    found_list = list(config_dir.glob("**/*.toml"))
    
    return found_list
    # -------------------------------------------------------------------------/



def get_batch_config_arg():
    """ using batch config mode
    """
    parser = argparse.ArgumentParser(description="using batch config mode")
    parser.add_argument(
        "--batch_mode",
        action="store_true",
    )
    
    args = parser.parse_args()
    
    return args
    # -------------------------------------------------------------------------/



def get_coupled_config_name(file:str):
    """
    """
    return f"{os.path.split(os.path.splitext(file)[0])[-1]}.toml"
    # -------------------------------------------------------------------------/