import os
import shutil
import filecmp
from pathlib import Path



def resave_result(original_path:Path, output_dir:Path, result_name:str):
    
    if isinstance(original_path, Path): original_path = str(original_path)
    else: raise TypeError("'original_path' should be a 'Path' object, please using `from pathlib import Path`")
    
    if not isinstance(output_dir, Path):
        raise TypeError("'output_dir' should be a 'Path' object, please using `from pathlib import Path`")
    
    if not isinstance(result_name, str):
        raise TypeError("'result_name' should be a 'str' object")
    
    if "MetaImage" in original_path: fish_name = original_path.split(os.sep)[-3]
    else: fish_name = original_path.split(os.sep)[-2]
    file_ext = result_name.split(".")[-1]
    out_path = output_dir.joinpath(f"{fish_name}.{file_ext}")
    shutil.copy(original_path, out_path)
    filecmp.cmp(original_path, out_path)