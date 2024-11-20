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
        Linux: "DejaVuSansMono.ttf"
    """
    platform_name = platform.system()
    if platform_name == "Windows":
        return "consola.ttf"
    elif platform_name == "Linux":
        return "DejaVuSansMono.ttf"
    else:
        raise NotImplementedError("Please assign a mono font on your system")
    # -------------------------------------------------------------------------/



def get_font(font_style: str=None, alt_default_family: str="sans-serif") -> Path:
    """

    Args:
        font_style (str, optional): one of `font` (name) on your system. Defaults to None.
        alt_default_family (str, optional): one of font family in `plt.rcParams`. Defaults to "sans-serif".
        
        NOTE: Without giving any argument, the font defaults to `DejaVu Sans` (matplotlib default).
    Raises:
        KeyError: if the given `alt_default_family` not exists.

    Returns:
        Path: a absolute font path on your system.
    """
    try:
        plt.rcParams[f"font.{alt_default_family}"]
        plt.rcParams["font.family"] = [alt_default_family]
    except KeyError:
        
        # get `font.family`
        mpl_font_family = []
        for k, v in plt.rcParams.items():
            if isinstance(v, list) and k.startswith("font."):
                if k == "font.family": continue
                mpl_font_family.extend([k.replace("font.", "")])

        raise KeyError(f"Expect `alt_default_family` is one of {mpl_font_family}")
    
    return Path(font_manager.findfont(font_style))
    # -------------------------------------------------------------------------/



def pt_to_px(pt: float, dpi: int=96) -> int:
    """ 1 pt = 1/72 inch
    """
    assert isinstance(pt, float), "param `pt` should be `float`"
    assert isinstance(dpi, int), "param `dpi` should be `int`"
    
    return round(pt * dpi / 72)
    # -------------------------------------------------------------------------/



def px_to_pt(px: int, dpi: int=96) -> float:
    """ 1 pt = 1/72 inch
    """
    assert isinstance(px, int), "param `px` should be `int`"
    assert isinstance(dpi, int), "param `dpi` should be `int`"
    
    return px * 72 / dpi
    # -------------------------------------------------------------------------/



def plot_with_imglist(img_list:List[np.ndarray], row:int, column:int, fig_dpi:int,
                      content:str, subtitle_list:Optional[List[str]]=None,
                      font_style:Optional[str]=None,
                      save_path:Optional[Path]=None, use_rgb:bool=False,
                      show_fig:bool=True, verbose:bool=False):
    """
    Show images in a gallery.

    Args:
        img_list (List[np.ndarray]): A list containing several images.
        row (int): The number of rows in the gallery.
        column (int): The number of columns in the gallery.
        fig_dpi (int): DPI for the figure.
        content (str): Informations to display beside the gallery.
        subtitle_list (Optional[List[str]], optional): Subtitles for each image. Defaults to None.
        font_style (Optional[str], optional): The **'absolute path'** to a font file. \
            If `None`, will use the first `sans-serif` font found by `matplotlib`. Defaults to None.
        save_path (Optional[Path], optional): The **'absolute path'** to save the figure. Defaults to None.
        use_rgb (bool, optional): Whether the images in `img_list` are in **'RGB'** order. Defaults to False.
        show_fig (bool, optional): Whether to display the gallery in a GUI window. Defaults to True.
        verbose (bool, optional): If True, will print debug information to the CLI. Defaults to False.
    """
    assert len(img_list) == (row*column), "len(img_list) != (row*column)"
    if subtitle_list is not None: assert len(subtitle_list) == len(img_list), "len(subtitle_list) != len(img_list)"
    
    # set default values
    if not font_style: font_style = str(get_font())
    
    # Get minimum image shape ( image may in different size )
    min_img_size = [np.inf, np.inf]
    for img in img_list:
        if img.shape[0] < min_img_size[0]: min_img_size[0] = img.shape[0]
        if img.shape[1] < min_img_size[1]: min_img_size[1] = img.shape[1]
    if min_img_size[0] < 256: min_img_size[0] = 256
    if min_img_size[1] < 256: min_img_size[1] = 256
    
    # Create figure
    # get `figsize`
    fig_w = (min_img_size[1]*column)
    fig_h = (min_img_size[0]*row)
    # init figure
    figsize = np.array((fig_w, fig_h))/plt.rcParams['figure.dpi']
    fig, axs = plt.subplots(row, column, figsize=figsize, dpi=fig_dpi)
    if verbose: print(f"Figure resolution : {figsize*fig_dpi}")
    
    # Plot each image
    for i, ax in enumerate(axs.flatten()):
        if use_rgb: img_rgb = img_list[i]
        else: img_rgb = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB) # BGR -> RGB
        ax.imshow(img_rgb, vmin=0, vmax=255)
        ax.set_axis_off()
    
    if subtitle_list is not None:
        # calculate font size for 'sub-titles'
        img_zoom_in = fig_dpi/plt.rcParams['figure.dpi']
        max_width = min_img_size[1]*img_zoom_in*0.95
        min_subtitle_px = round(min_img_size[0]*img_zoom_in*0.05) # iteration start
        for i, subtitle in enumerate(subtitle_list):
            if verbose: print(f"sub-titles: {i}, ", end="")
            opti_font_info = calculate_opti_title_param(subtitle, max_width,
                                                        min_subtitle_px,
                                                        font_style, verbose)
            # update `min_subtitle_px`
            if opti_font_info[-1] < min_subtitle_px:
                min_subtitle_px = opti_font_info[-1]
        
        min_subtitle_pt = px_to_pt(opti_font_info[-1], dpi=fig_dpi)
        if verbose: print(f"minimum 'sub-title' font size : {min_subtitle_px} px "
                          f"-> {min_subtitle_pt} pt")
        # add 'sub-titles'
        for i, ax in enumerate(axs.flatten()):
            ax.set_title(subtitle_list[i], fontsize=min_subtitle_pt)
    
    # auto layout
    fig.tight_layout()
    
    # add informations beside the gallery
    rgba_image = plt_to_pillow(fig) # matplotlib figure 預設為 RGBA (透明背景)
    content_font_size = round(min_img_size[0]*img_zoom_in*0.1)
    rgba_image = add_detail_info(rgba_image, content, font_size=content_font_size)
    
    if save_path is not None: rgba_image.save(save_path)
    if show_fig: rgba_image.show()
    
    rgba_image.close()
    plt.close(fig)
    # -------------------------------------------------------------------------/



def plot_with_imglist_auto_row(img_list:List[np.ndarray], column:int, fig_dpi:int,
                               content:str, subtitle_list:Optional[List[str]]=None,
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
    if verbose is True: print(f"( row, column ) = ( {auto_row}, {column} )")
    
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
        print(f"fontsize: {font_size}, "
              f"(title_width, title_height): ({title_width}, {title_height}), "
              f"target `max_width` = {max_width}")
    
    if title_width > max_width:
        return calculate_opti_title_param(title, max_width,
                                          int(0.9*font_size), font_style, verbose)
    
    if verbose: print("="*100, "\n")
    
    return title_width, title_height, font, font_size
    # -------------------------------------------------------------------------/



def plt_to_pillow(figure:figure.Figure):
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



def add_detail_info(rgba_image: Image.Image, content: str,
                    font_size: int, font_style: Optional[str]=None,
                    font_color: Optional[Tuple[int, int, int, int]]=None,
                    verbose: bool=False):
    """
    """
    # set default values (tuple color order: RGB)
    if not font_style: font_style = str(get_font(alt_default_family="monospace"))
    if not font_color: font_color = (0, 0, 0, 255)
    
    # create font object
    font = ImageFont.truetype(font_style, font_size)
    
    # get `width` and `height` of content
    image = Image.new("RGB", (0, 0), "white") # empty image
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), content, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # Create empty background in RGBA
    w_spacing = 0.05
    h_spacing = 0.05
    new_size = (int(rgba_image.width*(1+w_spacing*2) + width), int(rgba_image.height))
    new_canvas = Image.new('RGBA', new_size, color="#FFFFFF") # RGBA 可以建立透明背景
    if verbose: print(f'new_canvas.size {type(new_canvas.size)}: {new_canvas.size}')
    
    draw = ImageDraw.Draw(new_canvas)
    new_canvas.paste(rgba_image, (0, 0))
    draw.text((rgba_image.width*(1+w_spacing), rgba_image.height*h_spacing),
              content, font=font, fill=font_color)
    
    return new_canvas
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
    pred_text_width, pred_text_height = draw.textsize(pred_text, font=font)
    gt_text_width, gt_text_height = draw.textsize(gt_text, font=font)
    if pred_text_height == gt_text_height: text_height = gt_text_height
    
    # determine max width
    if pred_text_width > gt_text_width:
        text_width = pred_text_width
    else:
        text_width = gt_text_width
    
    gt_w = rgb_image.width - text_width - rgb_image.width*0.05
    gt_h = rgb_image.height - text_height - rgb_image.height*0.05
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
    darkratio_h = rgb_image.height - text_height - rgb_image.height*0.06
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