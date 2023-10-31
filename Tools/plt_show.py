import sys
from pathlib import Path
import argparse

abs_module_path = Path("./../").resolve()
if (abs_module_path.exists()) and (str(abs_module_path) not in sys.path):
    sys.path.append(str(abs_module_path)) # add path to scan customized module

from modules.plot.utils import plot_in_rgb, plot_in_gray, plot_by_channel
# -----------------------------------------------------------------------------/


def parse_args():
    
    parser = argparse.ArgumentParser(prog="plt_show", description="show images")
    parser.add_argument(
        "--img_path",
        type=str,
        help="Images to show."
    )
    gp_single_mode = parser.add_mutually_exclusive_group()
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
    # -------------------------------------------------------------------------/



if __name__ == "__main__":
    
    args = parse_args()
    
    if args.rgb:
        plot_in_rgb(args.img_path, (512, 512))
    elif args.gray:
        plot_in_gray(args.img_path, (512, 512))
    elif args.by_channel:
        plot_by_channel(args.img_path, (2048, 512))