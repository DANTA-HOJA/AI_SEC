import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from collections import Counter

from ...assert_fn import *
# -----------------------------------------------------------------------------/


def confusion_matrix_with_class(prediction:List[str], ground_truth:List[str]):
    """
    - confusion_matrix
    
    >>>  *  {0} {1} {2}  ~~> prediction
    >>> {0} [ ] [ ] [ ]  ~~> gt : 0
    >>> {1} [ ] [ ] [ ]  ~~> gt : 1
    >>> {2} [ ] [ ] [ ]  ~~> gt : 2
    """
    """ Count all classes in `prediction`, `ground_truth` """
    result_counter = Counter(ground_truth) + Counter(prediction)
    max_count = result_counter.most_common(1)[0][1] # usage: Counter.most_common(1) -> [('HD', 282)] <class 'list'> (回傳最大的前 x 項)
    all_class: list = sorted(list(result_counter.keys()))
    
    """ Create `confusion_matrix_list` """
    confusion_matrix_list: list = [" "] # 補 (0, 0) 空格
    confusion_matrix_list.extend(all_class) # 加上 column name
    for r_cls in all_class: # gt
        confusion_matrix_list.append(r_cls) # 加上 row name
        for c_cls in all_class: # pred
            match_cnt: int = 0
            for i in range(len(ground_truth)):
                if (prediction[i] == c_cls) and (ground_truth[i] == r_cls):
                    match_cnt += 1
            confusion_matrix_list.append(match_cnt)
    
    assert len(confusion_matrix_list) == ((len(all_class)+1)**2), "Failed to create 'confusion matrix' "
    
    """ Create `confusion_matrix_str` """
    confusion_matrix_str: str = ""
    confusion_matrix_str += "Confusion Matrix:\n\n"
    for i, item in enumerate(confusion_matrix_list):
        confusion_matrix_str += f"{item:>{len(str(max_count))+3}}"
        if (i+1)%(len(all_class)+1) == 0: confusion_matrix_str += "\n\n"

    return confusion_matrix_list, confusion_matrix_str
    # -------------------------------------------------------------------------/



def rename_history_dir(orig_history_dir:Path, test_method:str, model_state:str, test_log:dict):
    """
    """
    assert_is_pathobj(orig_history_dir)
    
    history_dir_split = str(orig_history_dir).split(os.sep)
    dir_name_split = re.split("{|}", history_dir_split[-1])
    
    new_name: str = ""
    new_name += f"{dir_name_split[0]}" # time_stamp
    new_name += f"{{{test_method}}}_"
    new_name += f"{{{dir_name_split[3]}}}_" # target_epochs_with_ImgLoadOptions
    new_name += f"{{{model_state}}}_"
    new_name += f"{{maweavg_f1_{test_log['maweavg_f1']}}}" 
    
    history_dir_split[-1] = new_name # replace `dir_name`
    new_history_dir = Path(os.sep.join(history_dir_split)) # reconstruct path
    os.rename(orig_history_dir, new_history_dir)
    # -------------------------------------------------------------------------/