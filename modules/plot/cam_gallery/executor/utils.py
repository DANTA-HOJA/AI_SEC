from typing import List, Dict, Tuple, Union
# -----------------------------------------------------------------------------/


def divide_fish_dsname_in_group(fish_dsname_list:List[str], worker:int) -> List[List[str]]:
    
    fish_dsname_list_group = []
    quotient  = int(len(fish_dsname_list)/(worker-1))
    for i in range((worker-1)):
        fish_dsname_list_group.append([ fish_dsname_list.pop(0) for i in range(quotient)])
    fish_dsname_list_group.append(fish_dsname_list)

    return fish_dsname_list_group
    # -------------------------------------------------------------------------/