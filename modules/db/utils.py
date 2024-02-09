from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

from ..assert_fn import assert_is_pathobj
# -----------------------------------------------------------------------------/


def flatten_dict(_dict:dict[str, Any], parent_key:str="", sep:str="."):
    """ flatten a dict. Concat all of keys become a "name chain" before it meet a value,
        return `new_dict = {"name chain": value}`

    Args:
        _dict (dict): a dictionary to flat
        parent_key (str, optional): key in previous level. Defaults to "".
        sep (str, optional): separator between keys. Defaults to ".".

    Returns:
        _type_: _description_
    """    
    new_dict: dict = {}
    
    for k, v in _dict.items():
        
        k = k.capitalize()
        flatten_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            new_dict.update(flatten_dict(v, flatten_key, sep=sep))
        else:
            if flatten_key in new_dict:
                raise KeyError(f"{flatten_key} already exist.")
            else:
                new_dict[flatten_key] = v
    
    return new_dict
    # -------------------------------------------------------------------------/



def create_path_hyperlink(display_text:str, path:Path) -> str:
    """
    """
    assert_is_pathobj(path)
    
    return '=HYPERLINK("file://{}", "{}")'.format(path, display_text)
    # -------------------------------------------------------------------------/



def resolve_path_hyperlink(excel_hyperlink:str) \
                            -> Union[tuple[None, None], tuple[str, Path]]:
    """
    """
    if isinstance(excel_hyperlink, str):
        if excel_hyperlink.startswith("=HYPERLINK("):
            str_split = excel_hyperlink.split('"')
            text = str_split[3]
            path = str_split[1].replace("file://", "")
            return text, Path(path)
    
    return None, None
    # -------------------------------------------------------------------------/