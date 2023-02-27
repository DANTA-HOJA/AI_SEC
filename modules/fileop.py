import os
from tqdm.auto import tqdm


def create_new_dir(path:str, end="\n", display_in_CLI=True, use_tqdm=False):
    if not os.path.exists(path):
        # if the demo_folder directory is not exist then create it.
        os.makedirs(path)
        if use_tqdm: tqdm.write(f"path: '{path}' is created!{end}")
        elif display_in_CLI: print(f"path: '{path}' is created!{end}")