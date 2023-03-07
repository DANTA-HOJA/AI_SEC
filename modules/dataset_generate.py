import os
import sys
import re
from typing import List, Dict, Tuple

from tqdm.auto import tqdm
import numpy as np
import cv2

sys.path.append(r"C:\Users\confocal_microscope\Desktop\ZebraFish_AP_POS\modules") # add path to scan customized module
from fileop import create_new_dir



def gen_dataset_name(xlsx_file:str, crop_size:int, crop_shift_region:str, intensity:int, drop_ratio:float, random_seed:int=None) -> str:
    """To generate dataset's name corresponing to the passing parameters.
    
    Args:
        xlsx_file (str):                e.g. "{4CLS_BY_SurfStDev}_data.xlsx" ---> SURF4C
        crop_size (int):                e.g.       512                       ---> CRPS512
        crop_shift_region (str):        e.g.      "1/4"                      ---> SF14
        intensity (int):                e.g.        20                       ---> INT20
        drop_ratio (float):             e.g.        0.3                      ---> DRP30
        random_seed (int, optional):    e.g.       2022                      ---> RS2022. Defaults to None.

    Raises:
        ValueError: "Numerator of 'crop_shift_region' needs to be 1"

    Returns:
        str:
        e.g. 'DS_SURF4C_CRPS512_SF14_INT20_DRP30' or 'DS_SURF4C_CRPS512_SF14_INT20_DRP30_RS2022'
    
    """
    
    # Converting... "xlsx_file" 
    xlsx_file_split = re.split("{|_|}", xlsx_file)
    n_class = xlsx_file_split[1].replace("CLS", "C")
    used_feature = xlsx_file_split[3].replace("StDev", "").upper()
    
    # Converting... "crop_shift_region"
    fraction = crop_shift_region.split("/")
    if int(fraction[0]) == 1: fraction = f"{fraction[0]}{fraction[1]}"
    else: raise ValueError("Numerator of 'crop_shift_region' needs to be 1")
    
    # Converting... "drop_ratio"
    ratio = int(drop_ratio*100)
     
    gen_name = f"DS_{used_feature}{n_class}_CRPS{str(crop_size)}_SF{fraction}_INT{str(intensity)}_DRP{str(ratio)}"
    
    if random_seed is None: return gen_name
    else: return f"{gen_name}_RS{str(random_seed)}"



def gen_crop_img(img:cv2.Mat, crop_size:int, shift_region:str) -> List:
    """generate the crop images using 'crop_size' and 'shift_region'

    Args:
        img (cv2.Mat): the image you want to generate its crop images
        crop_size (int): output size/shape of an crop image
        shift_region (str): if 'shift_region' is passed, calculate the shift region while creating each cropped image, e.g. shift_region=1/3, the overlap region of each cropped image is 2/3

    Returns:
        List
    """
    
    img_size = img.shape
    
    
    # *** calculate relation between img & crop_size ***
    
    # if 'shift_region' is passed, calculate the shift region while creating each cropped image, e.g. shift_region=1/3, the overlap region of each cropped image is 2/3
    fraction = shift_region.split("/")
    if int(fraction[0]) == 1: DIV_PIECES = int(fraction[1]) # DIV_PIECES, abbr. of 'divide pieces'
    else: raise ValueError("Numerator of 'shift_region' needs to be 1")
    # print(DIV_PIECES, "\n")
    # input()
    
    # Height
    quotient_h = int(img_size[0]/crop_size) # Height Quotient
    remainder_h = img_size[0]%crop_size # Height Remainder
    h_st_idx = remainder_h # image 只有上方有黑色
    h_crop_idxs = [(h_st_idx + int((crop_size/DIV_PIECES)*i)) 
                           for i in range(quotient_h*DIV_PIECES+1)]
    # print(f"h_crop_idxs = {h_crop_idxs}", f"len = {len(h_crop_idxs)}", "\n")
    # input()
    
    # Width
    quotient_w = int(img_size[1]/crop_size) # Width Quotient
    remainder_w = img_size[1]%crop_size # Width Remainder
    w_st_idx = int(remainder_w/2) # image 左右都有黑色，平分
    w_crop_idxs = [(w_st_idx + int((crop_size/DIV_PIECES)*i))
                           for i in range(quotient_w*DIV_PIECES+1)]
    # print(f"w_crop_idxs = {w_crop_idxs}", f"len = {len(w_crop_idxs)}", "\n")
    # input()
    
    
    # *** crop original image to small images ***
    
    # do crop 
    crop_img_list = []
    for i in range(len(h_crop_idxs)-DIV_PIECES):
        for j in range(len(w_crop_idxs)-DIV_PIECES):
            # print(h_crop_idxs[i], h_crop_idxs[i+DIV_PIECES], "\n", w_crop_idxs[j], w_crop_idxs[j+DIV_PIECES], "\n")
            crop_img_list.append(img[h_crop_idxs[i]:h_crop_idxs[i+DIV_PIECES], w_crop_idxs[j]:w_crop_idxs[j+DIV_PIECES], :])
    
    
    return crop_img_list



def drop_too_dark(crop_img_list:List[cv2.Mat], intensity:int, drop_ratio:float) -> Tuple[List[cv2.Mat], List[cv2.Mat]] : 
    """drop the image which too many dark pixels

    Args:
        crop_img_list (List[cv2.Mat]): a list contain 'gray images'
        intensity (int): a threshold (image is grayscale) to define too dark or not 
        drop_ratio (float): a threshold (pixel_too_dark / all_pixel) to decide the crop image keep or drop, if drop_ratio < 0.5, keeps the crop image.

    Returns:
        Tuple[List[cv2.Mat], List[cv2.Mat]]: first List is 'select_crop_img_list' , second List is 'drop_crop_img_list'
    """

    
    # check if the height and width of passed two lists with the same size
    img_size = crop_img_list[0].shape
    
    select_crop_img_list = []
    drop_crop_img_list = []

    
    for i in range(len(crop_img_list)):
        # change to grayscale
        in_grayscale = cv2.cvtColor(crop_img_list[i], cv2.COLOR_BGR2GRAY)
        # count pixels too dark
        pixel_too_dark = np.sum( in_grayscale < intensity )
        dark_ratio = pixel_too_dark/(img_size[0]*img_size[1])
        if dark_ratio < drop_ratio: # 有表皮資訊的
            select_crop_img_list.append(crop_img_list[i])
        else: 
            drop_crop_img_list.append(crop_img_list[i])


    return select_crop_img_list, drop_crop_img_list



def crop_img_saver(crop_img_list:List[cv2.Mat], crop_img_desc:str, save_dir:str, 
                   fish_size:str, fish_id:str, fish_pos:str, 
                   tqdm_process:tqdm, tqdm_overwrite_desc:str=None):
    
    
    # adjust settings of 'tqdm_process'
    if tqdm_overwrite_desc is not None: 
        tqdm_process.desc  = tqdm_overwrite_desc
    tqdm_process.n     = 0   # current value
    tqdm_process.total = len(crop_img_list)
    tqdm_process.refresh()
    
    # write crop images
    for j in range(len(crop_img_list)):
        save_dir_size = os.path.join(save_dir, crop_img_desc, fish_size)
        create_new_dir(save_dir_size, use_tqdm=True)
        write_name = f"{fish_size}_fish_{fish_id}_{fish_pos}_{crop_img_desc}_{j}.tiff"
        write_path = os.path.join(save_dir_size, write_name)

        crop_img = crop_img_list[j] # convenience to debug preview
        cv2.imwrite(write_path, crop_img)
        # cv2.imshow(write_name, select_crop_img)
        # cv2.waitKey(0)
        
        tqdm_process.update(1)
        tqdm_process.refresh()



def append_log(logs:List[Dict], fish_size:str, fish_id:str, fish_pos:str, all_class:List[str],
               crop_img_list:List[cv2.Mat], select_crop_img_list:List[cv2.Mat], drop_crop_img_list:List[cv2.Mat]):
    
    
    current_log = {
        "fish_name_comb": f"{fish_size}_fish_{fish_id}_{fish_pos}",
        #
        "number_of_crop": len(crop_img_list),
        "number_of_drop": len(drop_crop_img_list),
        "number_of_saved": len(select_crop_img_list),
    }
    #
    ## creat fish_size columns in current_log
    current_log["Class Count"] = ""
    for size in all_class: current_log[size] = 0
    #
    ## update # of saved images to current_log
    current_log[fish_size] = len(select_crop_img_list)
    #
    logs.append(current_log)
    # print current log in command
    # print(json.dumps(current_log, indent=2), "\n")