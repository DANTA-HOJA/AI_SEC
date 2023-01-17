import os
import sys
from datetime import datetime
import argparse
from copy import deepcopy

from typing import List, Dict

import json
import pandas as pd

import numpy as np
from collections import Counter

import cv2
import matplotlib.pyplot as plt

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
    def __init__(self, source_dir:str, img_names:List[str]):
        self.source_dir = source_dir
        self.img_names = img_names
        # self.class_map = class_map
        # self.num_classes = len(self.class_map)
        # self.label_in_filename = label_in_filename


    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, index):
        
        path = f"{self.source_dir}/{self.img_names[index]}"
        # print(path)

        # img preprocess
        img = cv2.imread(path)[:,:,::-1] #BGR -> RGB
        #
        # TODO:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) # resize to model input size:224
        #
        img = img / 255.0 # normalize to 0~1
        img = np.moveaxis(img, -1, 0) # img_info == 3: (H, W, C) -> (C, H, W)

        # # read class label
        # id = path.split(os.sep)[-1].split(".")[0]
        # cls = id.split("_")[self.label_in_filename]
        # cls_idx = self.class_map[cls]
        # # print(f"image[{index:^4d}] = {path}, class = {all_class[cls_idx]}")

        img = torch.from_numpy(img).float()  # 32-bit float
        # cls_idx = torch.tensor(cls_idx) # 64-bit int, can be [0] or [1] or [2] only
        
        return path, img #, cls_idx



def get_args():
    
    parser = argparse.ArgumentParser(description="zebrafish project: classification")
    parser.add_argument(
        "--cuda_idx",
        type=int,
        required=True,
        help="The index of cuda device.",
    )
    parser.add_argument(
        "--xlsx_file",
        type=str,
        required=True,
        help="The path of the Excel sheet.",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="The path of the dataset.",
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
        required=True,
        help="load model arch from torchvision, e.g. 'resnet50'.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path of model to load (file_ext = .ckpt, logs embedding).",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="the path to save the result Excel sheet.",
    )
        
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    args = get_args()
    
    
    # *** Variable ***
    # args variable
    cuda_idx = args.cuda_idx
    xlsx_file = args.xlsx_file
    source_dir = args.source_dir
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    save_path = args.save_path
    #
    # variable
    all_class = ["HD", "LD", "MD"]



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



    # *** Load Excel sheet as DataFrame(pandas) ***
    df_crop1l4Size = pd.read_excel(xlsx_file, engine = 'openpyxl')
    # print(df_crop1l4Size)
    df_names_list = df_crop1l4Size["image_name"].tolist()
    # print(type(df_names_list), len(df_names_list))



    # *** check data dir/file_name and show test image ***
    #
    # test_data
    print("\n", f"Anterior & Posterior, test_data ( total: {len(df_names_list)} )\n", "-"*100)
    [print(f"{i}：img_path = {df_names_list[i]}") for i in range(5)]
    # show images test
    test_path = f"{source_dir}/{df_names_list[-1]}"
    print("\n\n", "*** read test image ***")
    print(f" test_path = {test_path}\n\n")
    os.system("python plt_show.py --window_name {} --img_path {} --rgb".format("\"Read Test\"", f"\"{test_path}\""))



    # *** Create dataSets ***
    test_set = ImgDataset(source_dir=source_dir, img_names=df_names_list)
    assert test_set.__getitem__(-1)[0] == test_path, "path_A NOT match"


    # *** Initial dataLoader ***
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    
    
    # *** build model ***
    model_name = args.model_name
    model_name = model_name.replace("-", "_")
    # model = resnet50(pretrained=True)
    print(f"load model using 'torch.hub.load()', model_name = {model_name}", "\n\n")
    repo = 'pytorch/vision'
    model = torch.hub.load(repo, model_name, weights=None)
    #
    # print(model.fc) # Linear(in_features=2048, out_features=1000, bias=True)
    # modify fc for class = 3 (HD, LD, MD)
    model.fc = nn.Linear(in_features=2048, out_features=3, bias=True)
    #
    model.to(device)
    #
    # print(model)
    # os.system("pause")
    #
    # load model parameters
    ckpt_file = torch.load(args.model_path, device)
    # print(ckpt_file.keys(), "\n")
    model.load_state_dict(ckpt_file["model_state_dict"])
    
    
    
    # *** Testing: Anterior***
    # Testing variable
    all_pred_test_to_name = []
    # all_y_test_to_name = []
    # test_acc = 0.0
    #
    #    
    # *** Get start_time and Start Testing ***
    #
    model.eval() # set to evaluation mode
    with torch.no_grad(): 
        for i, data in enumerate(test_dataloader):
            _, x_test = data
            x_test = x_test.to(device)
            preds = model(x_test)
            _, pred_test = torch.max(preds, 1) # pred_dim = 3(class), pick up the max value in each class
            # test_acc += (pred_test.cpu() == y_test).sum().item()
            
            #show prediction vs. ground_truth in each batch

            pred_test_to_name = []
            # y_test_to_name = []
            
            for j in range(len(x_test)):
                pred_test_to_name.append(all_class[pred_test.cpu().numpy()[j]]) # type(pred_test.cpu().numpy()[j]) -> <class 'numpy.int64'>
                # y_test_to_name.append(all_class[y_test[j]])
                # print(pred_test_to_name, "\n")
                # print(y_test_to_name, "\n")
            
            pred_cnt = Counter(pred_test_to_name)
            print(f"Batch[{i+1}/{len(test_dataloader)}], {pred_cnt}")
            print("-"*100)
            
            
            all_pred_test_to_name.extend(pred_test_to_name)
            # all_y_test_to_name.extend(y_test_to_name)
            # print(len(all_pred_test_to_name), all_pred_test_to_name, "\n")
            # print(len(all_y_test_to_name), all_y_test_to_name, "\n")
    #
    assert len(all_pred_test_to_name) == len(df_names_list)
    
    
    
    # *** Update result to DataFrame ***
    df_crop1l4Size["preTest"] = all_pred_test_to_name
    print(df_crop1l4Size)
    #
    # save update Excel file to new path ***
    df_crop1l4Size.to_excel(save_path, index=False, engine = 'openpyxl')
    


    print("="*100, "\n", "process all complete !", "\n")