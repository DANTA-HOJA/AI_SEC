from glob import glob
import argparse

import json

import pandas as pd



def get_args():
    
    parser = argparse.ArgumentParser(description="zebrafish project: crop images into small pieces")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="The path of logs to read.",
    )
    parser.add_argument(
        "--orient",
        type=str,
        required=True,
        help="The parameter of 'pandas.read_json()'",
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = get_args()


    # *** Print CMD section divider ***
    print("="*100, "\n")



    # *** Read with open() ***
    #
    # with open(args.log_path, "r") as f:
    #     info = json.load(f)
    # df = pd.DataFrame(info)


    df = pd.read_json(args.log_path, orient=args.orient)


    print(f"Log Path: {args.log_path}\n")
    print(df, "\n")


    print("="*100, "\n", "process all complete !", "\n")