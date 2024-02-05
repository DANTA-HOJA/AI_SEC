import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

from ...assert_fn import *
from ...assert_fn import assert_0_or_1_history_dir
from ...shared.pathnavigator import PathNavigator
from ...shared.utils import exclude_tmp_paths
# -----------------------------------------------------------------------------/


def confusion_matrix_with_class(prediction:List[str], ground_truth:List[str]):
    """
    # Confusion Matrix
    
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



def rename_history_dir(orig_history_dir:Path, test_desc:str,
                       model_state:str, test_log:dict, score_key:str):
    """
    """
    assert_is_pathobj(orig_history_dir)
    
    history_dir_split = str(orig_history_dir).split(os.sep)
    dir_name_split = re.split("{|}", history_dir_split[-1])
    
    new_name: str = ""
    new_name += f"{dir_name_split[0]}" # time_stamp
    new_name += f"{{{test_desc}}}_"
    new_name += f"{{{dir_name_split[3]}}}_" # target_epochs_with_ImgLoadOptions
    new_name += f"{{{model_state}}}_"
    new_name += f"{{{score_key}_{test_log[f'{score_key}']}}}"
    
    history_dir_split[-1] = new_name # replace `dir_name`
    new_history_dir = Path(os.sep.join(history_dir_split)) # reconstruct path
    os.rename(orig_history_dir, new_history_dir)
    # -------------------------------------------------------------------------/



def reshape_transform(tensor, height=14, width=14):
    """ Control how many sub-images are needed to be divided.
        
        (This function is for `Vit_B_16 CAM Generator` only)
        
        - example :
            `patch_size` is `16` and `input_size` is `224`, 
            `(14 * 14)` sub-images will be generated.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    
    return result
    # -------------------------------------------------------------------------/



def get_history_dir(path_navigator:PathNavigator, time_stamp:str, state:str):
    """
    """
    if state not in ["best", "final"]:
        raise ValueError(f"(config) `model_prediction.state`: "
                            f"'{state}', accept 'best' or 'final' only\n")
    
    model_prediction: Path = \
        path_navigator.dbpp.get_one_of_dbpp_roots("model_prediction")
    best_found = []
    final_found = []
    
    # scan dir
    found_list = list(model_prediction.glob(f"**/{time_stamp}*"))
    found_list = exclude_tmp_paths(found_list)
    tmp_dict = {i: path for i, path in enumerate(found_list)}
    
    # assort dir
    for i, path in enumerate(found_list):
        if f"{{best}}" in str(path): best_found.append(tmp_dict.pop(i))
        elif f"{{final}}" in str(path): final_found.append(tmp_dict.pop(i))
    found_list = list(tmp_dict.values())
    
    # best mark
    if state == "best" and best_found:
        assert_0_or_1_history_dir(best_found, time_stamp, state)
        return best_found[0]
    
    # final mark
    if state == "final" and final_found:
        assert_0_or_1_history_dir(final_found, time_stamp, state)
        return final_found[0]
    
    # unset ( original )
    assert_0_or_1_history_dir(found_list, time_stamp, state)
    if found_list:
        return found_list[0]
    else:
        raise ValueError("No `history_dir` matches the provided config")
    # -------------------------------------------------------------------------/