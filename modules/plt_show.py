from typing import List, Tuple
import argparse
from math import floor

import numpy as np
import cv2
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



def plot_with_imglist(img_list:List[cv2.Mat],
                      fig_size:Tuple[float, float], row:int, column:int, 
                      fig_title:str, fig_title_font_size:int=26, 
                      subtitle:List[str]=None, subtitle_font_size:int=13, 
                      save_path:str=None, use_rgb:bool=False, show_fig:bool=True, plt=plt):
    
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
    
    # calculate 'figsize' # TODO:  Auto figure size
    fig_dpi = 100
    fig_size_div_dpi = []
    fig_size_div_dpi.append(fig_size[0]/fig_dpi)
    fig_size_div_dpi.append(fig_size[1]/fig_dpi)
    
    # Create figure
    fig, axs = plt.subplots(row, column, figsize=fig_size_div_dpi, dpi=fig_dpi)
    fig.suptitle(fig_title, fontsize=fig_title_font_size) # TODO:  Auto font size
    # plot each image
    if (row == 1) or (column == 1):
        
        for iter in range(row*column):
            if use_rgb: img_rgb = img_list[iter]
            else: img_rgb = cv2.cvtColor(img_list[iter], cv2.COLOR_BGR2RGB) # BGR -> RGB
            axs[iter].imshow(img_rgb, vmin=0, vmax=255)
            if subtitle is not None: axs[iter].set_title(subtitle[iter], fontdict={'fontsize': subtitle_font_size}) # TODO:  Auto font size
    
    else:
        
        for iter in range(row*column):
            i = floor(iter/column)
            j = floor(iter%column)
            # print(i, j)
            if use_rgb: img_rgb = img_list[iter]
            else: img_rgb = cv2.cvtColor(img_list[iter], cv2.COLOR_BGR2RGB) # BGR -> RGB
            axs[i, j].imshow(img_rgb, vmin=0, vmax=255)
            if subtitle is not None: axs[i, j].set_title(subtitle[iter], fontdict={'fontsize': subtitle_font_size}) # TODO:  Auto font size
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path is not None: fig.savefig(save_path)
    if show_fig: plt.show()
    
    plt.close()



def plot_with_imglist_auto_row(img_list:List[cv2.Mat], column:int, 
                               fig_title:str="", fig_title_font_size:int=26, 
                               subtitle:List[str]=None, subtitle_font_size:int=13, 
                               save_path:str=None, use_rgb:bool=False, 
                               verbose:bool=False, show_fig:bool=True, plt=plt):
    
    
    assert column <= len(img_list), f"len(img_list) = {len(img_list)}, but column = {column}, 'column' should not greater than 'len(img_list)'"
    
    # append empty arrays to the end of 'image_list' until its length is a multiple of 'column'
    orig_len = len(img_list)
    while len(img_list)%column != 0: img_list.append(np.ones_like(img_list[-1])*255)
    if verbose: print(f"len(img_list): {orig_len} --> {len(img_list)}")
    
    auto_row = int(len(img_list)/column)
    
    # calculate 'figsize' # TODO:  Auto figure size
    fig_w = (img_list[-1].shape[1])*column 
    fig_h = (img_list[-1].shape[0])*auto_row
    if verbose: print(f"figure resolution : {fig_w}, {fig_h}")
    
    # plot 
    kwargs_plot_with_imglist = {
        "img_list"             : img_list,
        "fig_title"            : " , ".join([fig_title, f"( row, column ) = ( {auto_row}, {column} )"]),
        "fig_title_font_size"  : fig_title_font_size,
        "fig_size"             : (fig_w, fig_h),
        "row"                  : auto_row,
        "column"               : column,
        "subtitle"             : subtitle,
        "subtitle_font_size"   : subtitle_font_size,
        "save_path"            : save_path,
        "use_rgb"              : use_rgb,
        "show_fig"             : show_fig,
    }
    plot_with_imglist(**kwargs_plot_with_imglist)



def parse_args():
    
    parser = argparse.ArgumentParser(prog="plt_show", description="show images")
    gp_single = parser.add_argument_group("single images")
    gp_single.add_argument(
        "--fig_title",
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