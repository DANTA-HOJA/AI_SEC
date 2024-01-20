import io
import os
import platform
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple  # Optional[] = Union[ , None]

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure, font_manager
from PIL import Image, ImageDraw, ImageFont
from pytorch_grad_cam.utils.image import show_cam_on_image
# -----------------------------------------------------------------------------/


def plot_in_rgb(img_path:str, fig_size:Tuple[float, float]):
    """ Show image in RGB color space.

    Args:
        img_path (str): read image using 'cv2.imread()'
        fig_size (Tuple[float, float]): pixel * pixel
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # calculate 'figsize'
    fig_dpi = 100
    fig_size_div_dpi = []
    fig_size_div_dpi.append(fig_size[0]/fig_dpi)
    fig_size_div_dpi.append(fig_size[1]/fig_dpi)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=fig_size_div_dpi, dpi=fig_dpi)
    fig.suptitle(f"Mode: RGB, img_size = {img_rgb.shape}")
    ax.imshow(img_rgb, vmin=0, vmax=255)
    
    plt.show()
    plt.close(fig)
    # -------------------------------------------------------------------------/



def plot_in_gray(img_path:str, fig_size:Tuple[float, float]):
    """ Show image in weighted-RGB gray scale.

    Args:
        img_path (str): read image using 'cv2.imread()'
        fig_size (Tuple[float, float]): pixel * pixel
    """
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # calculate 'figsize'
    fig_dpi = 100
    fig_size_div_dpi = []
    fig_size_div_dpi.append(fig_size[0]/fig_dpi)
    fig_size_div_dpi.append(fig_size[1]/fig_dpi)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=fig_size_div_dpi, dpi=fig_dpi)
    fig.suptitle(f"Mode: Gray (weighted-RGB), img_size = {img_gray.shape}")
    ax.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    
    plt.show()
    plt.close(fig)
    # -------------------------------------------------------------------------/



def plot_by_channel(img_path:str, fig_size:Tuple[float, float]):
    """ Show image by splitting its channels.

    Args:
        img_path (str): read image using 'cv2.imread()'
        fig_size (Tuple[float, float]): pixel * pixel
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    ch_R, ch_G, ch_B = cv2.split(img_rgb) # split channel
    
    # calculate 'figsize'
    fig_dpi = 100
    fig_size_div_dpi = []
    fig_size_div_dpi.append(fig_size[0]/fig_dpi)
    fig_size_div_dpi.append(fig_size[1]/fig_dpi)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=fig_size_div_dpi, dpi=fig_dpi)
    fig.suptitle(f"Mode: RGB_Channel, img_size = {img_rgb.shape}")
    
    # merge RGB (original)
    axes[0].set_title("RGB")
    axes[0].imshow(img_rgb, vmin=0, vmax=255)
    
    # plot R Channel
    axes[1].set_title("R")
    axes[1].imshow(ch_R, cmap='gray', vmin=0, vmax=255)
    
    # plot G Channel
    axes[2].set_title("G")
    axes[2].imshow(ch_G, cmap='gray', vmin=0, vmax=255)
    
    # plot B Channel
    axes[3].set_title("B")
    axes[3].imshow(ch_B, cmap='gray', vmin=0, vmax=255)
    
    plt.show()
    plt.close(fig)
    # -------------------------------------------------------------------------/



def get_matplotlib_default_font():
    """ `matplotlib` default font: `DejaVu Sans`
    """
    return font_manager.findfont(plt.rcParams['font.sans-serif'][0])
    # -------------------------------------------------------------------------/



def get_mono_font():
    """ Windows: "consola.ttf"
        Linux: "UbuntuMono-R.ttf"
    """
    platform_name = platform.system()
    if platform_name == "Windows":
        return "consola.ttf"
    elif platform_name == "Linux":
        return "UbuntuMono-R.ttf"
    else:
        raise NotImplementedError("Please assign a mono font on your system")
    # -------------------------------------------------------------------------/



def plot_with_imglist(img_list:List[np.ndarray], row:int, column:int, fig_dpi:int,
                      figtitle:str, subtitle_list:Optional[List[str]]=None,
                      font_style:Optional[str]=None,
                      save_path:Optional[Path]=None, use_rgb:bool=False,
                      show_fig:bool=True, verbose:bool=False):
    """ Show images in gallery way

    Args:
        img_list (List[np.ndarray]): a list contain several images, channel_order = BGR (default of 'cv2.imread()')
        row (int): number of rows of gallery.
        column (int): number of columns of gallery.
        fig_dpi (int): argumnet of matplotlib figure.
        figtitle (str, optional): big title of gallery.
        subtitle_list (Optional[List[str]], optional): title of each sub images. Defaults to None.
        font_style (Optional[str], optional): if None will use matplotlib default font. Defaults to None.
        save_path (Optional[Path], optional): Defaults to None.
        use_rgb (bool, optional): specify images in `img_list` are `RGB` order images. Defaults to False.
        show_fig (bool, optional): show gallery on GUI window. Defaults to True.
        verbose (bool, optional): if True, will show debug info on CLI output. Defaults to False.
    """
    assert len(img_list) == (row*column), "len(img_list) != (row*column)"
    if subtitle_list is not None: assert len(subtitle_list) == len(img_list), "len(subtitle_list) != len(img_list)"
    
    # Get the path of matplotlib default font: `DejaVu Sans`
    if not font_style: font_style = get_matplotlib_default_font()
    
    # Get minimum image shape ( image may in different size )
    min_img_shape = [np.inf, np.inf]
    for img in img_list:
        if img.shape[0] < min_img_shape[0]: min_img_shape[0] = img.shape[0]
        if img.shape[1] < min_img_shape[1]: min_img_shape[1] = img.shape[1]
    
    # Calculate auto `figsize`
    fig_w = min_img_shape[1]*column/100 # `100` is defalut value of `dpi` in `plt.figure()`
    fig_h = min_img_shape[0]*row/100 # `100` is defalut value of `dpi` in `plt.figure()`
    if verbose: print(f"figure resolution : {fig_w*fig_dpi}, {fig_h*fig_dpi}")
    
    # Create figure
    fig, axs = plt.subplots(row, column, figsize=(fig_w, fig_h), dpi=fig_dpi)
    fig.tight_layout() # auto layout
    
    # Plot each image
    for i, ax in enumerate(axs.flatten()):
        if use_rgb: img_rgb = img_list[i]
        else: img_rgb = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB) # BGR -> RGB
        ax.imshow(img_rgb, vmin=0, vmax=255)
    
    if subtitle_list is not None:
        for i, ax in enumerate(axs.flatten()):
            font_size = int(min_img_shape[0]*0.05)
            max_width = min_img_shape[1]*0.91*0.75
            opti_font_info = calculate_opti_title_param(subtitle_list[i], max_width,
                                                        font_size, font_style, verbose)
            ax.set_title(subtitle_list[i], fontsize=opti_font_info[3]) # TODO:  find method to optimize `fontsize` of `subtitle`
    
    # Calculate space occupied by yaxis and ylabel
    bbox = ax.yaxis.get_tightbbox(fig.canvas.get_renderer())
    y_width, y_height = bbox.width, bbox.height
    
    # Store figure into `buffer`
    rgba_image = plt_to_pillow(fig, os.path.dirname(save_path)) # matplotlib figure 預設為 RGBA (透明背景)
    
    # Draw `title` on `background`
    rgba_image = add_big_title(rgba_image, figtitle, ylabel_width=y_width, 
                               font_style=font_style, verbose=verbose)
    
    if save_path is not None: rgba_image.save(save_path)
    if show_fig: rgba_image.show()
    
    rgba_image.close()
    plt.close(fig)
    # -------------------------------------------------------------------------/



def plot_with_imglist_auto_row(img_list:List[np.ndarray], column:int, fig_dpi:int,
                               figtitle:str, subtitle_list:Optional[List[str]]=None,
                               font_style:Optional[str]=None,
                               save_path:Optional[Path]=None, use_rgb:bool=False,
                               show_fig:bool=True, verbose:bool=False):
    """
    """
    assert column <= len(img_list), f"len(img_list) = {len(img_list)}, but column = {column}, 'column' should not greater than 'len(img_list)'"
    if subtitle_list is not None: assert len(subtitle_list) == len(img_list), "len(subtitle_list) != len(img_list)"
    
    # append empty arrays to the end of 'image_list' until its length is a multiple of 'column'
    orig_len = len(img_list)
    
    if (len(img_list)%column != 0) and (subtitle_list is None):
        subtitle_list = [ "" for _ in range(len(img_list)) ]
    while len(img_list)%column != 0:
        img_list.append(np.ones_like(img_list[-1])*255)
        subtitle_list.append(f"[Empty]")
    if verbose: print(f"len(img_list): {orig_len} --> {len(img_list)}")
    
    input_args = locals() # collect all exist local variables (before this line) as a dict
    input_args.pop("orig_len")
    
    auto_row = int(len(img_list)/column)
    input_args["row"] = auto_row
    input_args["figtitle"] = " , ".join([input_args["figtitle"], f"( row, column ) = ( {auto_row}, {column} )"])
    
    # plot
    plot_with_imglist(**input_args)
    # -------------------------------------------------------------------------/



def calculate_opti_title_param(title:str, max_width:int,
                               font_size:int, font_style:Optional[str]=None,
                               verbose:bool=False):
    """
    """
    # set default values
    if not font_style: font_style = get_mono_font()
    
    # text
    font = ImageFont.truetype(font_style, font_size)

    # Get title size
    title_bbox = font.getbbox(title) # (left, top, right, bottom) bounding box
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    
    if verbose: 
        print(f'fontsize: {font_size}, (title_width, title_height): ({title_width}, {title_height})')
    
    if title_width > max_width:
        return calculate_opti_title_param(title, max_width,
                                          int(0.9*font_size), font_style, verbose)
    
    if verbose: print("="*100, "\n")
    return title_width, title_height, font, font_size
    # -------------------------------------------------------------------------/



def plt_to_pillow(figure:figure.Figure, temp_file_dir:Path):
    """
    """
    # size_mb = 50
    # initial_bytes = b"\0" * size_mb * 1024 * 1024
    # buffer = io.BytesIO(initial_bytes=initial_bytes)
    # figure.savefig(buffer, format='png')
    # buffer.seek(0)
    # pil_img = deepcopy(Image.open(buffer)) # 轉換成 PIL Image
    #                                        ## Note: matplotlib figure 預設為 RGBA (透明背景)
    # buffer.close()
    
    with tempfile.NamedTemporaryFile(suffix='.png') as f_writer:
        figure.savefig(f_writer, format='png')
        f_writer.seek(0)
        pil_img = deepcopy(Image.open(f_writer))
    
    return pil_img
    # -------------------------------------------------------------------------/



def add_big_title(rgba_image:Image.Image, title:str, title_line_height:int=2,
                  font_style:Optional[str]=None, font_color: Optional[Tuple[int, int, int, int]]=None,
                  ylabel_width:int=0, verbose:bool=False):
    """
    """
    # set default values (tuple color order: RGB)
    if not font_style: font_style = get_matplotlib_default_font()
    if not font_color: font_color = (0, 0, 0, 255)
    
    # Get title parameters
    font_size = int(rgba_image.height*0.05)
    max_width = rgba_image.width*0.95
    title_width, title_height, font, _ = \
        calculate_opti_title_param(title, max_width,
                                   font_size, font_style, verbose)
    title_space = int(title_height*title_line_height) # title + line height

    # Create empty background in RGBA
    background_size = (rgba_image.width, rgba_image.height + title_space)
    # background = Image.new('RGB', (pil_img_rgba.width, pil_img_rgba.height+100), color = (255, 255, 255))
    background = Image.new('RGBA', background_size, color = (0, 0, 0, 0)) # RGBA 可以建立透明背景
    if verbose: print(f'background.size {type(background.size)}: {background.size}')
    
    # init draw component
    draw = ImageDraw.Draw(background)

    # Put the `rgba_image` with offset `title_space`
    background.paste(rgba_image, (0, title_space))

    # Draw `title` on `background`
    # width center posistion = Simplify[ ((background.width - ylabel_width) - title_width)/2 + ylabel_width ]
    title_width_center = (background.width - title_width)/2 + 0.5*ylabel_width
    draw.text((title_width_center, 0), title, font=font, fill=font_color)
    
    return background
    # -------------------------------------------------------------------------/



def draw_x_on_image(rgb_image:Image.Image,
                    line_color:Optional[Tuple[int, int, int]]=None,
                    line_width:Optional[int]=None):
    """
    """
    # set default values (tuple color order: RGB)
    if not line_color: line_color = (180, 160, 0)
    if not line_width: line_width = 2
    
    # init draw component
    draw = ImageDraw.Draw(rgb_image)
    
    # set 4 corners
    top_left = (0, 0)
    top_right = (rgb_image.width, 0)
    bottom_left = (0, rgb_image.height)
    bottom_right = (rgb_image.width, rgb_image.height)

    # draw 2 diagonal lines
    draw.line((top_left, bottom_right), fill=tuple(line_color), width=line_width)
    draw.line((top_right, bottom_left), fill=tuple(line_color), width=line_width)
    # -------------------------------------------------------------------------/



def draw_predict_ans_on_image(rgb_image:Image.Image, pred_cls:str, gt_cls:str,
                              font_style:Optional[str]=None, font_size:Optional[int]=None,
                              correct_color:Optional[Tuple[int, int, int]]=None,
                              incorrect_color:Optional[Tuple[int, int, int]]=None,
                              shadow_color:Optional[Tuple[int, int, int]]=None):
    """
    """
    # set default values (tuple color order: RGB)
    if not font_style: font_style = get_mono_font()
    ## auto `fontsize` = image_height * 0.07
    if not font_size: font_size = round(np.array(rgb_image).shape[0]*0.07)
    if not correct_color: correct_color = (0, 255, 0)
    if not incorrect_color: incorrect_color = (255, 255, 255)
    if not shadow_color: shadow_color = (0, 0, 0)
    
    # init draw component
    draw = ImageDraw.Draw(rgb_image)
    
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
    # -------------------------------------------------------------------------/



def draw_drop_info_on_image(rgb_image:Image.Image, intensity:int, dark_ratio:float, drop_ratio:float, 
                            font_style:Optional[str]=None, font_size:Optional[int]=None,
                            selected_color:Optional[Tuple[int, int, int]]=None,
                            drop_color:Optional[Tuple[int, int, int]]=None,
                            shadow_color:Optional[Tuple[int, int, int]]=None):
    """
    """
    # set default values (tuple color order: RGB)
    if not font_style: font_style = get_mono_font()
    ## auto `fontsize` = image_height * 0.05
    if not font_size: font_size = round(np.array(rgb_image).shape[0]*0.05)
    if not selected_color: selected_color = (255, 255, 255)
    if not drop_color: drop_color = (255, 255, 127)
    if not shadow_color: shadow_color = (0, 0, 0)
    
    # init draw component
    draw = ImageDraw.Draw(rgb_image)
    
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
    # -------------------------------------------------------------------------/



def postprocess_cam_image(image:np.ndarray, grayscale_cam:np.ndarray, use_rgb,
                          colormap:int, image_weight:float, cam_save_path:str,
                          pred_cls:str, gt_cls:str, resize:Optional[Tuple[int, int]]=None,
                          font_style:Optional[str]=None, font_size:Optional[int]=None):
    """
    """
    # set default values
    if not font_style: font_style = get_mono_font()
    if not font_size: font_size = 16
    
    if use_rgb: rgb_img = image
    else: rgb_img = image[:, :, ::-1] # BGR -> RGB
    
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, True, colormap, image_weight)
    if resize is not None:
        cam_image = cv2.resize(cam_image, resize, interpolation=cv2.INTER_CUBIC)
    
    cam_image = Image.fromarray(cam_image)
    if pred_cls != gt_cls: draw_predict_ans_on_image(cam_image, pred_cls, gt_cls, font_style, font_size)
    
    cv2.imwrite(cam_save_path, cv2.cvtColor(np.array(cam_image), cv2.COLOR_RGB2BGR))
    # -------------------------------------------------------------------------/