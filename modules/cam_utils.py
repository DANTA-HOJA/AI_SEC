import os
import numpy as np

import cv2
from PIL import Image, ImageDraw, ImageFont

from pytorch_grad_cam.utils.image import show_cam_on_image


# ref: https://github.com/jacobgil/pytorch-grad-cam


def reshape_transform(tensor, height=14, width=14):
    
    # Control how many sub-images are needed to be divided.
    #   When 'patch_size' is 16 and 'input_size' is 224,
    #   (14 * 14) sub-images will be generated.
    # 
    # ref: https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html#how-does-it-work-with-vision-transformers
    
    result = tensor[:, 1:, :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def add_text_on_cam_image(image:np.ndarray, pred_cls:str, gt_cls:str, font_size:int) -> np.ndarray:
    
    img = Image.fromarray(image)
    font = ImageFont.truetype("consola.ttf", font_size)
    draw = ImageDraw.Draw(img)
    
    # text
    pred_text = f"{'prediction '} : {pred_cls}"
    gt_text   = f"{'groundtruth'} : {gt_cls}"
    
    # text color
    if gt_cls == pred_cls: text_color = (0, 255, 0) # RGB
    else: text_color = (255, 0, 0) # RGB
    
    # calculate text position
    text_width, text_height = draw.textsize(gt_text, font=font)
    gt_w = img.width - text_width - img.width*0.05
    gt_h = img.height  - text_height - img.height*0.05
    gt_position = [gt_w, gt_h]
    pred_position = [gt_w, (gt_h - font_size*1.5)]
    
    # shadow
    shadow_color = (0, 0, 0)
    shadow_offset = (2, 2)

    # pred text
    draw.text((pred_position[0] + shadow_offset[0], pred_position[1] + shadow_offset[1]), 
              pred_text, font=font, fill=shadow_color) # shadow
    draw.text(pred_position, pred_text, font=font, fill=text_color)
    
    # gt text
    draw.text((gt_position[0] + shadow_offset[0], gt_position[1] + shadow_offset[1]), 
              gt_text, font=font, fill=shadow_color) # shadow
    draw.text(gt_position, gt_text, font=font, fill=text_color)
    
    return np.array(img)



def postprocess_cam_image(rgb_img:np.ndarray, grayscale_cam:np.ndarray, resolution:int,
                          pred_cls:str, gt_cls:str, font_size:int, cam_save_path:str):
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = cv2.resize(cam_image, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
    
    if pred_cls != gt_cls: cam_image = add_text_on_cam_image(cam_image, pred_cls, gt_cls, font_size)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(cam_save_path, cam_image)
    
