import os
import sys
import traceback
from datetime import datetime

from glob import glob

import argparse
from pathlib import Path
from typing import List, Dict

from tqdm.auto import tqdm

import json
import pandas as pd
import numpy as np
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import plt_show

import torch
from torch import nn, utils
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet50, resnext50_32x4d

from sklearn.metrics import classification_report



def create_new_dir(path:str, end="\n"):
    if not os.path.exists(path):
        # if the demo_folder directory is not exist then create it.
        os.makedirs(path)
        print(f"path: '{path}' is created!{end}")



def confusion_matrix(ground_truth:List, prediction:List):
    
    # Count all possible class in ground_truth, prediction
    result_counter = Counter(ground_truth) + Counter(prediction)
    max_count = result_counter.most_common(1)[0][1] # result_counter.most_common(1) -> [('HD', 282)] <class 'list'> (回傳最大的前 x 項)
    all_class = sorted(list(result_counter.keys()))
    
    
    # Create "confusion matrix"
    confusion_matrix_list = [" "] # 補 (0, 0) 空格
    confusion_matrix_list.extend(all_class) # 加上 column name
    for r_cls in all_class:
        confusion_matrix_list.append(r_cls) # 加上 row name
        for c_cls in all_class:
            match_cnt = 0
            for i in range(len(ground_truth)):
                if (ground_truth[i] == r_cls) and (prediction[i] == c_cls): match_cnt += 1
            confusion_matrix_list.append(match_cnt)
    
    assert len(confusion_matrix_list) == ((len(all_class)+1)**2), 'Create "confusion matrix" failed'
    
    
    # Show in command
    print("Confusion Matrix:\n")
    for enum, item in enumerate(confusion_matrix_list):
        print(f"{item:>{len(str(max_count))+3}}", end="")
        if (enum+1)%(len(all_class)+1) == 0: print("\n")
    print("\n", end="")


    return confusion_matrix_list



class ImgDataset(Dataset):
    def __init__(self, paths:List[str], class_map:Dict[str, int], label_in_filename:int):
        self.paths = paths
        self.class_map = class_map
        self.num_classes = len(self.class_map)
        self.label_in_filename = label_in_filename


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        path = self.paths[index]

        # img preprocess
        img = cv2.imread(path)[:,:,::-1] #BGR -> RGB
        #
        # TODO:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) # resize to model input size:224
        #
        img = img / 255.0 # normalize to 0~1
        img = np.moveaxis(img, -1, 0) # img_info == 3: (H, W, C) -> (C, H, W)

        # read class label
        id = path.split(os.sep)[-1].split(".")[0]
        cls = id.split("_")[self.label_in_filename]
        cls_idx = self.class_map[cls]
        # print(f"image[{index:^4d}] = {path}, class = {all_class[cls_idx]}")

        img = torch.from_numpy(img).float()  # 32-bit float
        cls_idx = torch.tensor(cls_idx) # 64-bit int, can be [0] or [1] or [2] only
        
        return img, cls_idx



def get_args():
    
    parser = argparse.ArgumentParser(description="zebrafish project: classification")
    parser.add_argument(
        "--cuda_idx",
        type=int,
        required=True,
        help="The index of cuda device.",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="The path of the dataset.",
    )
    parser.add_argument(
        "--n_class",
        type=int,
        default=3,
        help="Total classes in the given dataset",
    )
    parser.add_argument(
        "--label_in_filename",
        type=int,
        default=0,
        help="The index of label/class in the filename separate by '_'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="training args: batch_size.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="model name/architecture.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path of model to load (file_ext = .ckpt, logs embedding).",
    )
    
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    args = get_args()
    
    
    # *** Variable ***
    # args variable
    cuda_idx = args.cuda_idx
    source_dir = args.source_dir
    n_class = args.n_class
    label_in_filename = args.label_in_filename
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    #
    # variable
    #



    # *** GPU settings ***
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.set_device(cuda_idx)
    #
    device_num = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_num)
    #
    print("="*100, "\n")
    print(f"Using {device}")
    print(f"device_num = {device_num}")
    print(f"device_name = {device_name}", "\n")



    # *** Get dir_name as class/label name ***
    all_class = []
    for file_or_dir in os.listdir(source_dir):
        if os.path.isdir(f"{source_dir}/{file_or_dir}"):
            # print(file_or_dir)
            all_class.append(file_or_dir)
    class_map = {cls:i for i,cls in enumerate(all_class)}
    print(f"class_map = {class_map}\n")



    # *** Get image path ***
    fish_paths = glob(f"{source_dir}/*/*.tiff")
    print(f"total = {len(fish_paths)}")
    #
    # # NOTE: Debug: only select first 200
    # fish_paths = fish_paths[:200]
    # print(f"Debug mode, only select first {len(fish_paths)}\n")



    # *** Check data dir/file_name and show test image ***
    # test_data
    fish_path_test = fish_paths
    print("\n", f"test_data ( total: {len(fish_path_test)} )\n", "-"*100)
    [print(f"{i}：img_path = {fish_path_test[i]}") for i in range(5)]
    # show images test
    path = fish_path_test[-1]
    print("\n\n", "*** read test image ***")
    print(f" path = {path}\n\n")
    os.system("python plt_show.py --window_name {} --img_path {} --rgb".format("\"Test Image\"", path))



    # *** Create dataSets ***
    test_set = ImgDataset(fish_path_test, class_map=class_map, label_in_filename=args.label_in_filename)



    # *** Initial dataLoader ***
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)



    # *** Build model ***
    model_name_rp = model_name.replace("-", "_")
    print(f"load model using 'torch.hub.load()', model_name = {model_name_rp}", "\n\n")
    repo = 'pytorch/vision'
    model = torch.hub.load(repo, model_name_rp, weights=None)
    #
    # print(model.heads.head) # vit_b_16.heads.head --> Linear(in_features=768, out_features=1000, bias=True)
    #
    # modify fc for class = 3 (HD, LD, MD)
    model.heads.head = nn.Linear(in_features=768, out_features=n_class, bias=True)
    #
    # print(model)
    #
    model.to(device)
    #
    # load model parameters
    ckpt_file = torch.load(model_path, device)
    # print(ckpt_file.keys(), "\n")
    model.load_state_dict(ckpt_file["model_state_dict"])



    # *** Testing ***
    # testing variable
    all_pred_test_to_name = []
    all_y_test_to_name = []
    test_acc = 0.0
    #
    #
    #
    # *** get start_time and start testing ***
    model.eval() # set to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            x_test, y_test = data
            x_test = x_test.to(device)
            preds = model(x_test)
            _, pred_test = torch.max(preds, 1) # pred_dim = 3(class), pick up the max value in each class
            test_acc += (pred_test.cpu() == y_test).sum().item()
            
            
            # record prediction vs. ground_truth in each batch

            pred_test_to_name = []
            y_test_to_name = []
            
            for j in range(len(y_test)):
                pred_test_to_name.append(all_class[pred_test.cpu().numpy()[j]]) # type(pred_test.cpu().numpy()[j]) -> <class 'numpy.int64'>
                y_test_to_name.append(all_class[y_test[j]])
            # print(pred_test_to_name, "\n")
            # print(y_test_to_name, "\n")
            
            
            print(f"Batch[{i+1}/{len(test_dataloader)}], # of (ground truth == prediction) in_this_batch： {(pred_test.cpu() == y_test).sum().item()}/{len(y_test)}")
            print("-"*70)
            
            
            all_pred_test_to_name.extend(pred_test_to_name)
            all_y_test_to_name.extend(y_test_to_name)
            # print(len(all_pred_test_to_name), all_pred_test_to_name, "\n")
            # print(len(all_y_test_to_name), all_y_test_to_name, "\n")



    # *** Calculate final accuracy ***
    time_stamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    final_acc = test_acc / len(test_set)
    print(f"Time: {time_stamp}, Accuray = {final_acc:.4f}\n")



    # *** Classification Report ***
    cls_report = classification_report(all_y_test_to_name, all_pred_test_to_name)
    print("Classification Report:\n\n", cls_report, "\n")



    # *** Confusion Matrix ***
    #   row: Ground truth
    #   column: predict
    #  *　0　1　2
    #  0 [] [] []
    #  1 [] [] []
    #  2 [] [] []
    #
    confusion_mat = confusion_matrix(all_y_test_to_name, all_pred_test_to_name)


    print("="*100, "\n", "process all complete !\n")