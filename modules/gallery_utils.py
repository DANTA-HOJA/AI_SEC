import os
import sys
from typing import List, Dict, Tuple
from copy import deepcopy

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from pytorch_grad_cam.utils.image import show_cam_on_image



def draw_x_on_image(rgb_image:Image.Image, color:Tuple[int, int], line_width:int):
    
    draw = ImageDraw.Draw(rgb_image)
    
    # set 4 corners
    top_left = (0, 0)
    top_right = (rgb_image.width, 0)
    bottom_left = (0, rgb_image.height)
    bottom_right = (rgb_image.width, rgb_image.height)

    # draw 2 diagonal lines
    draw.line((top_left, bottom_right), fill=color, width=line_width) # RGB
    draw.line((top_right, bottom_left), fill=color, width=line_width) # RGB


    
def draw_predict_ans_on_image(rgb_image:Image.Image, pred_cls:str, gt_cls:str,
                     font_style:str="consola.ttf", font_size:int=None):
    
    draw = ImageDraw.Draw(rgb_image)
    
    # text
    pred_text = f"prediction : {pred_cls}"
    gt_text   = f"groundtruth: {gt_cls}"
    if font_size is not None: font_size=font_size
    else: font_size = round(np.array(rgb_image).shape[0]*0.07) # auto_font_size = image_height * 0.07
    font = ImageFont.truetype(font_style, font_size)
    
    # text color
    if gt_cls == pred_cls: text_color = (0, 255, 0) # color: RGB
    else: text_color = (255, 255, 255) # color: RGB
    
    # calculate text position
    text_width, text_height = draw.textsize(gt_text, font=font)
    gt_w = rgb_image.width - text_width - rgb_image.width*0.05
    gt_h = rgb_image.height  - text_height - rgb_image.height*0.05
    gt_pos = [gt_w, gt_h]
    pred_pos = [gt_w, (gt_h - font_size*1.5)]
    
    # shadow
    shadow_color = (0, 0, 0)
    shadow_offset = (2, 2)

    # draw 'prediction' text
    draw.text((pred_pos[0] + shadow_offset[0], pred_pos[1] + shadow_offset[1]), 
               pred_text, font=font, fill=shadow_color) # shadow
    draw.text(pred_pos, pred_text, font=font, fill=text_color)
    
    # draw 'groundtruth' text
    draw.text((gt_pos[0] + shadow_offset[0], gt_pos[1] + shadow_offset[1]), 
               gt_text, font=font, fill=shadow_color) # shadow
    draw.text(gt_pos, gt_text, font=font, fill=text_color)



def draw_drop_info_on_image(rgb_image:Image.Image, intensity:int, dark_ratio:float, drop_ratio:float, 
                           font_style:str="consola.ttf", font_size:int=None):
    
    draw = ImageDraw.Draw(rgb_image)

    # text
    intensity_text = f"@ intensity: {intensity}"
    darkratio_text = f">> dark_ratio: {dark_ratio:.5f}"
    if font_size is not None: font_size=font_size
    else: font_size = round(np.array(rgb_image).shape[0]*0.05) # auto_font_size = image_height * 0.05
    font = ImageFont.truetype(font_style, font_size)

    # text color
    if dark_ratio >= drop_ratio: text_color = (255, 255, 127) # drop, color: RGB
    else: text_color = (255, 255, 255) # selected, color: RGB
    
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
    
    # shadow
    shadow_color = (0, 0, 0)
    shadow_offset = (2, 2)
    
    # draw 'intensity' text
    draw.text((intensity_pos[0] + shadow_offset[0], intensity_pos[1] + shadow_offset[1]), 
               intensity_text, font=font, fill=shadow_color) # shadow
    draw.text(intensity_pos, intensity_text, font=font, fill=text_color)
    
    # draw 'dark_ratio' text
    draw.text((darkratio_pos[0] + shadow_offset[0], darkratio_pos[1] + shadow_offset[1]), 
               darkratio_text, font=font, fill=shadow_color) # shadow
    draw.text(darkratio_pos, darkratio_text, font=font, fill=text_color)



def postprocess_cam_image(image:np.ndarray, grayscale_cam:np.ndarray, use_rgb,
                          colormap:int, image_weight:float, cam_save_path:str,
                          pred_cls:str, gt_cls:str, resize:Tuple[int, int]=None,
                          font_style:str="consola.ttf", font_size:int=None):
    
    if use_rgb: rgb_img = image
    else: rgb_img = image[:, :, ::-1] # BGR -> RGB
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, True, colormap, image_weight)
    if resize is not None:
        cam_image = cv2.resize(cam_image, resize, interpolation=cv2.INTER_CUBIC)
    
    cam_image = Image.fromarray(cam_image)
    if pred_cls != gt_cls: draw_predict_ans_on_image(cam_image, pred_cls, gt_cls, font_style, font_size)
    
    cv2.imwrite(cam_save_path, cv2.cvtColor(np.array(cam_image), cv2.COLOR_RGB2BGR))