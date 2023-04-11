from typing import List, Tuple
import argparse

import numpy as np
import cv2
from PIL import ImageFont

from matplotlib import font_manager
import matplotlib.pyplot as plt



def plot_in_rgb(img_path:str, fig_size:Tuple[float, float], plt=plt):
    
    """
    show image in RGB color space.
    
    Args:
        window_name (str): GUI_window/figure name.
        img ( cv2.Mat ): an image you want to display, channel orient = BGR (default orient of 'cv2.imread()')
        plt (module): matplotlib.pyplot.
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # calculate 'figsize'
    fig_dpi = 100
    fig_size_div_dpi = []
    fig_size_div_dpi.append(fig_size[0]/fig_dpi)
    fig_size_div_dpi.append(fig_size[1]/fig_dpi)
    
    # Create figure
    fig = plt.figure(figsize=fig_size_div_dpi, dpi=fig_dpi)
    fig.suptitle(f"Channel: ' RGB ' , shape = {img_rgb.shape}")
    plt.imshow(img_rgb, vmin=0, vmax=255)
    plt.show()
    plt.close()



def plot_in_gray(img_path:str, fig_size:Tuple[float, float], plt=plt):
    
    """
    show image in weighted-RGB gray scale.
    
    Args:
        window_name (str): GUI_window/figure name.
        img ( cv2.Mat ): an image you want to display, channel orient = BGR (default orient of 'cv2.imread()')
        plt (module): matplotlib.pyplot.
    """
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # calculate 'figsize'
    fig_dpi = 100
    fig_size_div_dpi = []
    fig_size_div_dpi.append(fig_size[0]/fig_dpi)
    fig_size_div_dpi.append(fig_size[1]/fig_dpi)
    
    # Create figure
    fig = plt.figure(figsize=fig_size_div_dpi, dpi=fig_dpi)
    fig.suptitle(f"Channel: ' Gray (weighted-RGB) ' , shape = {img_gray.shape}")
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=255)
    
    plt.show()
    plt.close()



def plot_by_channel(img_path:str, fig_size:Tuple[float, float], plt=plt):
    
    """
    show an BGR image by splitting its channels.
    
    Args:
        window_name (str): GUI_window/figure name.
        img ( cv2.Mat ): an image you want to display, channel orient = BGR (default orient of 'cv2.imread()')
        plt (module): matplotlib.pyplot.
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
    fig, axs = plt.subplots(1, 4, figsize=fig_size_div_dpi, dpi=fig_dpi)
    fig.suptitle(f"Plot by ' RGB Channel ' , shape = {img_rgb.shape}")
    
    # merge RGB (original)
    axs[0].set_title(f"Merge RGB")
    axs[0].imshow(img_rgb, vmin=0, vmax=255)
    
    # plot R Channel
    axs[1].set_title("R")
    axs[1].imshow(ch_R, cmap='gray', vmin=0, vmax=255)
    
    # plot G Channel
    axs[2].set_title("G")
    axs[2].imshow(ch_G, cmap='gray', vmin=0, vmax=255)
    
    # plot B Channel
    axs[3].set_title("B")
    axs[3].imshow(ch_B, cmap='gray', vmin=0, vmax=255)
    
    plt.show()
    plt.close()



def plot_with_imglist(img_list:List[cv2.Mat], row:int, column:int, fig_dpi:int,
                      figtitle:str, subtitle:List[str]=None, subtitle_fontsize:int=13,
                      save_path:str=None, use_rgb:bool=False,
                      show_fig:bool=True, verbose:bool=False,
                      plt_default_font:str=None):
    
    """
    show an RGB image by splitting its channels.
    
    Args: [  TODO:  not update yet ]
        window_name (str): GUI_window/figure name.
        img_list ( List[cv2.Mat] ): an list contain several images, channel orient = BGR (default orient of 'cv2.imread()')
        row (int): number of rows in GUI_window.
        column (int): number of columns in GUI_window.
        title (list): optional, title for each figure.
        plt (module): matplotlib.pyplot.
    """
    
    assert len(img_list) == (row*column), "len(img_list) != (row*column)"
    
    # Get the path of matplotlib default font: `DejaVu Sans`
    if plt_default_font is None:
        plt_default_font=font_manager.findfont(plt.rcParams['font.sans-serif'][0])
    
    # Get minimum image shape
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
    
    # Calculate auto `fontsize`
    figtitle_fontsize = round(fig_h*fig_dpi*0.02)
    font = ImageFont.truetype(plt_default_font, size=figtitle_fontsize)
    text_width, text_height = font.getsize(figtitle)
    ## if `text_width` is too long, keep searching a proper font size
    while text_width > (fig_w*fig_dpi)*0.7:
        figtitle_fontsize = round(figtitle_fontsize*0.9)
        font = ImageFont.truetype(plt_default_font, size=figtitle_fontsize)
        text_width, text_height = font.getsize(figtitle)
        if verbose: print((f"(text_width, text_height) = ({text_width}, {text_height}), " 
                           f"auto font size = {figtitle_fontsize}"))
    
    # Calculate the ratio between `text_height` and `fig_height`
    title_h_ratio = text_height/(fig_h*fig_dpi)
    if verbose: print(f"text_height = {text_height}, ratio = {title_h_ratio}")
    
    # Plot figure title
    fig.text(0.5, (1-title_h_ratio), figtitle, fontsize=figtitle_fontsize, ha='center', va='bottom')
    
    # Plot each image
    for i, ax in enumerate(axs.flatten()):
        
        if use_rgb: img_rgb = img_list[i]
        else: img_rgb = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB) # BGR -> RGB
        ax.imshow(img_rgb, vmin=0, vmax=255)
        if subtitle is not None: ax.set_title(subtitle[i], fontdict={'fontsize': subtitle_fontsize}) # TODO:  optimize set_title() with `subtitle_fontsize`
    
    # Adjust figure layout
    fig.tight_layout()
    plt.subplots_adjust(top=(1-title_h_ratio*2))
    
    if save_path is not None: fig.savefig(save_path)
    if show_fig: plt.show()
    
    plt.close()



def plot_with_imglist_auto_row(img_list:List[cv2.Mat], column:int, fig_dpi:int,
                               figtitle:str="", subtitle:List[str]=None, subtitle_fontsize:int=13,
                               save_path:str=None, use_rgb:bool=False,
                               show_fig:bool=True, verbose:bool=False,
                               plt_default_font:str=None):
    
    
    assert column <= len(img_list), f"len(img_list) = {len(img_list)}, but column = {column}, 'column' should not greater than 'len(img_list)'"
    
    # Get the path of matplotlib default font: `DejaVu Sans`
    if plt_default_font is None:
        plt_default_font=font_manager.findfont(plt.rcParams['font.sans-serif'][0])
    
    input_args = locals() # collect all exist local variables (before this line) as a dict
    
    # append empty arrays to the end of 'image_list' until its length is a multiple of 'column'
    orig_len = len(img_list)
    while len(img_list)%column != 0: img_list.append(np.ones_like(img_list[-1])*255)
    if verbose: print(f"len(img_list): {orig_len} --> {len(img_list)}")
    
    auto_row = int(len(img_list)/column)
    
    input_args["row"] = auto_row
    input_args["figtitle"] = " , ".join([input_args["figtitle"], f"( row, column ) = ( {auto_row}, {column} )"])
    input_args["subtitle_fontsize"] = 13 # TODO:  optimize set_title() with `subtitle_fontsize`
    
    # plot
    plot_with_imglist(**input_args)



def parse_args():
    
    parser = argparse.ArgumentParser(prog="plt_show", description="show images")
    gp_single = parser.add_argument_group("single images")
    gp_single.add_argument(
        "--figtitle",
        type=str,
        help="the BIG title of figure."
    )
    gp_single.add_argument(
        "--img_path",
        type=str,
        help="Images to show."
    )
    gp_single_mode = gp_single.add_mutually_exclusive_group()
    gp_single_mode.add_argument(
        "--rgb",
        action="store_true",
        help="Show images in RGB.",
    )
    gp_single_mode.add_argument(
        "--gray",
        action="store_true",
        help="Show images in GRAY.",
    )
    gp_single_mode.add_argument(
        "--by_channel",
        action="store_true",
        help="Show images by RGB channel.",
    )
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    args = parse_args()
    
    if args.rgb:
        plot_in_rgb(args.window_name, args.img_path, plt)
    elif args.gray:
        plot_in_gray(args.window_name, args.img_path, plt)
    elif args.by_channel:
        plot_by_channel(args.window_name, args.img_path, plt)