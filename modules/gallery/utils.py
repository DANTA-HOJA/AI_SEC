import os
import sys
from typing import List, Dict, Tuple, Optional # Optional[] = Union[ , None]
from copy import deepcopy

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from pytorch_grad_cam.utils.image import show_cam_on_image



def divide_in_grop(fish_dsname_list:List[str], worker:int) -> List[List[str]]:
    
    fish_dsname_list_group = []
    quotient  = int(len(fish_dsname_list)/(worker-1))
    for i in range((worker-1)):
        fish_dsname_list_group.append([ fish_dsname_list.pop(0) for i in range(quotient)])
    fish_dsname_list_group.append(fish_dsname_list)

    return fish_dsname_list_group



def draw_x_on_image(rgb_image:Image.Image,
                    line_color:Optional[Tuple[int, int, int]]=None, line_width:Optional[int]=None):
    
    draw = ImageDraw.Draw(rgb_image)
    
    # set default value, color: RGB
    if not line_color: line_color = (180, 160, 0)
    if not line_width: line_width = 2
    
    # set 4 corners
    top_left = (0, 0)
    top_right = (rgb_image.width, 0)
    bottom_left = (0, rgb_image.height)
    bottom_right = (rgb_image.width, rgb_image.height)

    # draw 2 diagonal lines
    draw.line((top_left, bottom_right), fill=tuple(line_color), width=line_width)
    draw.line((top_right, bottom_left), fill=tuple(line_color), width=line_width)


    
def draw_predict_ans_on_image(rgb_image:Image.Image, pred_cls:str, gt_cls:str,
                              font_style:Optional[str]=None, font_size:Optional[int]=None,
                              correct_color:Optional[Tuple[int, int, int]]=None,
                              incorrect_color:Optional[Tuple[int, int, int]]=None,
                              shadow_color:Optional[Tuple[int, int, int]]=None):
    
    draw = ImageDraw.Draw(rgb_image)
    
    # set default value, color: RGB
    if not font_style: font_style = "consola.ttf"
    ## auto `fontsize` = image_height * 0.07
    if not font_size: font_size = round(np.array(rgb_image).shape[0]*0.07)
    if not correct_color: correct_color = (0, 255, 0)
    if not incorrect_color: incorrect_color = (255, 255, 255)
    if not shadow_color: shadow_color = (0, 0, 0)
    
    # text
    pred_text = f"prediction : {pred_cls}"
    gt_text   = f"groundtruth: {gt_cls}"
    font = ImageFont.truetype(font_style, font_size)
    
    # text color
    if gt_cls == pred_cls: text_color = correct_color
    else: text_color = incorrect_color
    
    # calculate text position
    text_width, text_height = draw.textsize(gt_text, font=font)
    gt_w = rgb_image.width - text_width - rgb_image.width*0.05
    gt_h = rgb_image.height  - text_height - rgb_image.height*0.05
    gt_pos = [gt_w, gt_h]
    pred_pos = [gt_w, (gt_h - font_size*1.5)]
    
    # shadow, stroke (text border)
    shadow_offset = (2, 2)
    stroke_width = 2

    # draw 'prediction' text
    draw.text((pred_pos[0] + shadow_offset[0], pred_pos[1] + shadow_offset[1]), 
               pred_text, font=font, fill=tuple(shadow_color),
               stroke_width=stroke_width, stroke_fill=tuple(shadow_color)) # shadow
    draw.text(tuple(pred_pos), pred_text, font=font, fill=tuple(text_color))
    
    # draw 'groundtruth' text
    draw.text((gt_pos[0] + shadow_offset[0], gt_pos[1] + shadow_offset[1]), 
               gt_text, font=font, fill=tuple(shadow_color),
               stroke_width=stroke_width, stroke_fill=tuple(shadow_color)) # shadow
    draw.text(tuple(gt_pos), gt_text, font=font, fill=tuple(text_color))



def draw_drop_info_on_image(rgb_image:Image.Image, intensity:int, dark_ratio:float, drop_ratio:float, 
                            font_style:Optional[str]=None, font_size:Optional[int]=None,
                            selected_color:Optional[Tuple[int, int, int]]=None,
                            drop_color:Optional[Tuple[int, int, int]]=None,
                            shadow_color:Optional[Tuple[int, int, int]]=None):
    
    draw = ImageDraw.Draw(rgb_image)
    
    # set default value, color: RGB
    if not font_style: font_style = "consola.ttf"
    ## auto `fontsize` = image_height * 0.05
    if not font_size: font_size = round(np.array(rgb_image).shape[0]*0.05)
    if not selected_color: selected_color = (255, 255, 255)
    if not drop_color: drop_color = (255, 255, 127)
    if not shadow_color: shadow_color = (0, 0, 0)

    # text
    intensity_text = f"@ intensity: {intensity}"
    darkratio_text = f">> dark_ratio: {dark_ratio:.5f}"
    font = ImageFont.truetype(font_style, font_size)

    # text color
    if dark_ratio >= drop_ratio: text_color = drop_color
    else: text_color = selected_color
    
    # calculate text position
    ## dark_ratio
    text_width, text_height = draw.textsize(darkratio_text, font=font)
    darkratio_w = (rgb_image.width - text_width)/2
    darkratio_h = rgb_image.height  - text_height - rgb_image.height*0.06
    darkratio_pos = [darkratio_w, darkratio_h]
    ## intensity
    text_width, text_height = draw.textsize(intensity_text, font=font)
    intensity_w = (rgb_image.width - text_width)/2
    intensity_h = (darkratio_h - font_size*1.5)
    intensity_pos = [intensity_w, intensity_h]
    
    # shadow, stroke (text border)
    shadow_offset = (2, 2)
    stroke_width = 2
    
    # draw 'intensity' text
    draw.text((intensity_pos[0] + shadow_offset[0], intensity_pos[1] + shadow_offset[1]), 
               intensity_text, font=font, fill=tuple(shadow_color), 
               stroke_width=stroke_width, stroke_fill=tuple(shadow_color)) # shadow
    draw.text(tuple(intensity_pos), intensity_text, font=font, fill=tuple(text_color))
    
    # draw 'dark_ratio' text
    draw.text((darkratio_pos[0] + shadow_offset[0], darkratio_pos[1] + shadow_offset[1]), 
               darkratio_text, font=font, fill=tuple(shadow_color), 
               stroke_width=stroke_width, stroke_fill=tuple(shadow_color)) # shadow
    draw.text(tuple(darkratio_pos), darkratio_text, font=font, fill=tuple(text_color))



def postprocess_cam_image(image:np.ndarray, grayscale_cam:np.ndarray, use_rgb,
                          colormap:int, image_weight:float, cam_save_path:str,
                          pred_cls:str, gt_cls:str, resize:Optional[Tuple[int, int]]=None,
                          font_style:Optional[str]=None, font_size:Optional[int]=None):
    
    if not font_style: font_style = "consola.ttf"
    if not font_size: font_size = 16
    
    if use_rgb: rgb_img = image
    else: rgb_img = image[:, :, ::-1] # BGR -> RGB
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, True, colormap, image_weight)
    if resize is not None:
        cam_image = cv2.resize(cam_image, resize, interpolation=cv2.INTER_CUBIC)
    
    cam_image = Image.fromarray(cam_image)
    if pred_cls != gt_cls: draw_predict_ans_on_image(cam_image, pred_cls, gt_cls, font_style, font_size)
    
    cv2.imwrite(cam_save_path, cv2.cvtColor(np.array(cam_image), cv2.COLOR_RGB2BGR))