# TODO: 
# -----------------------------------------------------------------------------/


def flatten_dict(_dict:dict[str, any], parent_key:str="", sep:str="."):
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