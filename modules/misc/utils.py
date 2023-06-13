from typing import List

def get_target_str_idx_in_list(source_list:List[str], target_str:str):
    
    target_idx = None
    
    for i, text in enumerate(source_list):
        if target_str in text:
            if target_idx is None:
                target_idx = i
            else:
                raise ValueError(f"Too many '{target_str}' in list")
    
    return target_idx