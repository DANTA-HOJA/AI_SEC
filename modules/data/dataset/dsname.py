import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
# -----------------------------------------------------------------------------/


def get_dsname_sortinfo(string_with_fish_dsname:Union[str, Path]) -> tuple:
    """
    """
    """ Type check """
    if isinstance(string_with_fish_dsname, Path): string_with_fish_dsname = str(string_with_fish_dsname)
    elif isinstance(string_with_fish_dsname, str): pass
    else: raise TypeError("Unrecognized type of `string_with_fish_dsname`. Only `pathlib.Path` or `str` are accepted.")
    
    fish_dsname = string_with_fish_dsname.split(os.sep)[-1]
    fish_dsname = os.path.splitext(fish_dsname)[0] # [fish_dsname, tiff]
    fish_dsname_split = re.split(" |_|-", fish_dsname)
    
    if len(fish_dsname_split) == 3:
        """ fish_228_A  -->  ['fish', '228', 'A'] """
        return int(fish_dsname_split[1]), fish_dsname_split[2]
    elif len(fish_dsname_split) == 4:
        """ fish_228_A_D  -->  ['fish', '228', 'A', 'D'] """
        return int(fish_dsname_split[1]), fish_dsname_split[2], fish_dsname_split[3]
    elif len(fish_dsname_split) == 5:
        """ fish_228_A_crop_0  -->  ['fish', '13', 'A', 'crop', '00'] """
        return int(fish_dsname_split[1]), fish_dsname_split[2], int(fish_dsname_split[4])
    elif len(fish_dsname_split) == 6:
        """ fish_228_A_D_crop_0  -->  ['fish', '228', 'A', 'D', 'crop', '00'] """
        return int(fish_dsname_split[1]), fish_dsname_split[2], fish_dsname_split[3], \
               int(fish_dsname_split[5])
    else:
        raise NotImplementedError(f"Unrecognized format of 'dsname' : {fish_dsname}")
    # -------------------------------------------------------------------------/