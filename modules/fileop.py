import os
import filecmp
from tqdm.auto import tqdm



def create_new_dir(path:str, end="\n", display_in_CLI=True, use_tqdm=False):
    if not os.path.exists(path):
        # if the demo_folder directory is not exist then create it.
        os.makedirs(path)
        if use_tqdm: tqdm.write(f"path: '{path}' is created!{end}")
        elif display_in_CLI: print(f"path: '{path}' is created!{end}")



def resave_result(original_path:str, output_dir:str, result_path:str):
    if "MetaImage" in original_path: fish_name = original_path.split(os.sep)[-3]
    else: fish_name = original_path.split(os.sep)[-2]
    file_ext = result_path.split(".")[-1]
    out_path = os.path.join(output_dir, f"{fish_name}.{file_ext}")
    os.system(f"copy \"{original_path}\" \"{out_path}\" ")
    filecmp.cmp(original_path, out_path)