from typing import List, Tuple
import argparse
from math import floor

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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    axs[2].imshow(ch_R, cmap='gray', vmin=0, vmax=255)
    
    # plot B Channel
    axs[3].set_title("B")
    axs[3].imshow(ch_R, cmap='gray', vmin=0, vmax=255)
    
    plt.show()
    plt.close()



def plot_with_imglist(img_list:List[cv2.Mat],
                      fig_title:str, fig_size:Tuple[float, float], 
                      row:int, column:int, subtitle:List[str]=None, plt=plt):
    
    """
    show an RGB image by splitting its channels.
    
    Args:
        window_name (str): GUI_window/figure name.
        img_list ( List[cv2.Mat] ): an list contain several images, channel orient = BGR (default orient of 'cv2.imread()')
        row (int): number of rows in GUI_window.
        column (int): number of columns in GUI_window.
        title (list): optional, title for each figure.
        plt (module): matplotlib.pyplot.
    """
    
    assert len(img_list) == (row*column), "len(img_list) != (row*column)"
    
    # calculate 'figsize'
    fig_dpi = 100
    fig_size_div_dpi = []
    fig_size_div_dpi.append(fig_size[0]/fig_dpi)
    fig_size_div_dpi.append(fig_size[1]/fig_dpi)
    
    # Create figure
    fig, axs = plt.subplots(row, column, figsize=fig_size_div_dpi, dpi=fig_dpi)
    fig.suptitle(fig_title)
    # plot each image
    for iter in range(row*column):
        i = floor(iter/column)
        j = floor(iter%column)
        # print(i, j)
        img_rgb = cv2.cvtColor(img_list[iter], cv2.COLOR_BGR2RGB) # BGR -> RGB
        axs[i, j].imshow(img_rgb, vmin=0, vmax=255)
        if subtitle is not None: axs[i, j].set_title(subtitle[iter])
    
    plt.show()
    plt.close()



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