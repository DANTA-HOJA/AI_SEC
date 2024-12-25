from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
# -----------------------------------------------------------------------------/


def get_seg_desc(config: dict) -> str:
    """
    """
    seg_desc = config["seg_results"]["seg_desc"]
    accept_str = ["SLIC", "Cellpose"]
    if seg_desc not in accept_str:
        raise ValueError(f"`config.seg_desc`, only accept {accept_str}\n")
    
    return seg_desc
    # -------------------------------------------------------------------------/


def parse_base_size(config: dict) -> tuple[int, int]:
    """
    """
    base_size: str = config["seg_results"]["base_size"]
    size: list[str] = base_size.split("_")
    
    # width
    size_w = int(size[0].replace("W", ""))
    assert size_w <= 512, "Maximum support `width` is 512"
    
    # height
    size_h = int(size[1].replace("H", ""))
    assert size_h <= 1024, "Maximum support `height` is 1024"
    
    return size_w, size_h
    # -------------------------------------------------------------------------/


def get_slic_param_name(config: dict) -> str:
    """
    """
    n_segments: int  = config["SLIC"]["n_segments"]
    dark: int        = config["SLIC"]["dark"]
    merge: int       = config["SLIC"]["merge"]
    
    return f"S{n_segments}_D{dark}_M{merge}"
    # -------------------------------------------------------------------------/


def get_cellpose_param_name(config: dict) -> str:
    """
    """
    cp_model_name: str = config["Cellpose"]["cp_model_name"]
    channels: list     = config["Cellpose"]["channels"]
    merge: int         = config["Cellpose"]["merge"]
    
    tmp_list = []
    tmp_list.append(cp_model_name)
    tmp_list.append(f"CH{channels[0]}{channels[1]}")
    tmp_list.append(f"M{merge}")
    
    return "_".join(tmp_list)
    # -------------------------------------------------------------------------/


def get_class_weight_dict(dataset_df:pd.DataFrame,
                          labels: list, label2idx: dict[str, int]):
    """
    """
    counter = Counter(dataset_df["class"])
    
    # rearrange dict
    class_counts_dict: dict[str, int] = {}
    for cls in labels:
        class_counts_dict[cls] = counter[cls]
    
    class_weights_dict: dict[int, int] = {}
    total_samples = sum(class_counts_dict.values())
    
    for key, value in class_counts_dict.items(): # value = number of samples of the class
        class_weights_dict[label2idx[key]] = (1 - (value/total_samples))
    
    return class_weights_dict
    # -------------------------------------------------------------------------/


def save_confusion_matrix_display(y_true: list[str],
                                  y_pred: list[str],
                                  save_path: Path,
                                  feature_desc: str,
                                  dataset_desc: str):
    """
    save 2 file:
    1. `save_path`/`feature_desc`.cm.png
    2. `save_path`/`feature_desc`.cm.normgt.png
    """
    ConfusionMatrixDisplay.from_predictions(y_true=y_true,
                                            y_pred=y_pred)
    plt.tight_layout()
    plt.ylabel("Ground truth")
    plt.xlabel("Prediction")
    plt.savefig(save_path.joinpath(f"{feature_desc}.{dataset_desc}.cm.png"))
    plt.savefig(save_path.joinpath(f"{feature_desc}.{dataset_desc}.cm.svg"))
    
    # normalized by row (summation of ground truth samples)
    ConfusionMatrixDisplay.from_predictions(y_true=y_true,
                                            y_pred=y_pred,
                                            normalize="true",
                                            im_kw={"vmin":0.0, "vmax":1.0})
    plt.tight_layout()
    plt.ylabel("Ground truth")
    plt.xlabel("Prediction")
    plt.savefig(save_path.joinpath(f"{feature_desc}.{dataset_desc}.cm.normgt.png"))
    plt.savefig(save_path.joinpath(f"{feature_desc}.{dataset_desc}.cm.normgt.svg"))
    # -------------------------------------------------------------------------/