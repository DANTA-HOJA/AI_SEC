from typing import List
import argparse

import cv2
import matplotlib.pyplot as plt



def img_rgb(window_name:str, img_path:str, plt=plt):
    
    """
    show image in RGB color space.
    
    Args:
        window_name (str): GUI_window/figure name.
        img ( cv2.Mat ): an image you want to display, channel orient = BGR (default orient of 'cv2.imread()')
        plt (module): matplotlib.pyplot.
    """
    
    # window_name
    fig = plt.figure(window_name)
    img = cv2.imread(img_path)
    image_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.title(f"channel = RGB, shape = {image_RGB.shape}")
    plt.imshow(image_RGB, vmin=0, vmax=255)
    plt.show()
    
   
    
def img_gray(window_name:str, img_path:str, plt=plt):
    
    """
    show image in RGB gray scale.
    
    Args:
        window_name (str): GUI_window/figure name.
        img ( cv2.Mat ): an image you want to display, channel orient = BGR (default orient of 'cv2.imread()')
        plt (module): matplotlib.pyplot.
    """
    
    # window_name
    fig = plt.figure(window_name)
    img = cv2.imread(img_path)
    image_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.title(f"Gray, shape = {image_GRAY.shape}")
    plt.imshow(image_GRAY, cmap='gray', vmin=0, vmax=255)
    plt.show()



def img_by_channel(window_name:str, img_path:str, plt=plt):
    
    """
    show an BGR image by splitting its channels.
    
    Args:
        window_name (str): GUI_window/figure name.
        img ( cv2.Mat ): an image you want to display, channel orient = BGR (default orient of 'cv2.imread()')
        plt (module): matplotlib.pyplot.
    """
    
    # BGR -> RGB
    img = cv2.imread(img_path)
    image_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dpi = 100
    fig_W = (image_RGB.shape[0]/dpi)*4-5
    fig_H = (image_RGB.shape[1]/dpi)
    
    # window_name
    fig = plt.figure(window_name, figsize=(fig_W, fig_H))
    
    # split channel
    B_ch, G_ch, R_ch = cv2.split(img)
    ax_141 = fig.add_subplot(1, 4, 1)
    ax_141.set_title(f"merge, shape = {image_RGB.shape}")
    plt.imshow(image_RGB, vmin=0, vmax=255)
    
    # plot R Chnnel
    ax_142 = fig.add_subplot(1, 4, 2)
    ax_142.set_title("R")
    plt.imshow(R_ch, cmap='gray', vmin=0, vmax=255)
    
    # plot G Chnnel
    ax_143 = fig.add_subplot(1, 4, 3)
    ax_143.set_title("G")
    plt.imshow(G_ch, cmap='gray', vmin=0, vmax=255)
    
    # plot B Chnnel
    ax_144 = fig.add_subplot(1, 4, 4)
    ax_144.set_title("B")
    plt.imshow(B_ch, cmap='gray', vmin=0, vmax=255)
    
    plt.show()



def img_list(window_name:str, img_list:List[cv2.Mat], row:int, column:int, title:List[str]=None, plt=plt):
    
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
    
    # window_name
    fig = plt.figure(window_name)
    
    
    for i in range(row*column):

        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB)
        fig.add_subplot(row, column, i+1)
        if title is not None:
            plt.title(title[i])
        plt.imshow(img_rgb, vmin=0, vmax=255)
    
    plt.show()
    

   
def parse_args():
    
    parser = argparse.ArgumentParser(prog="plt_show", description="show images")
    gp_single = parser.add_argument_group("single images")
    gp_single.add_argument(
        "--window_name",
        type=str,
        help="window name."
    )
    gp_single.add_argument(
        "--img_path",
        type=str,
        help="Images to shown."
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
        img_rgb(args.window_name, args.img_path, plt)
    elif args.gray:
        img_gray(args.window_name, args.img_path, plt)
    elif args.by_channel:
        img_by_channel(args.window_name, args.img_path, plt)