from ....data.dataset.utils import parse_dataset_file_name
# -----------------------------------------------------------------------------/


def get_gallery_column(dataset_base_size: str,
                       dataset_file_name: str) -> int:
    """
    """
    base_width = dataset_base_size.split("_")[0]
    base_width = int(base_width.replace("W", ""))
    tmp_dict = parse_dataset_file_name(dataset_file_name)
    
    crop_size: int = tmp_dict["crop_size"]
    pieces = int(tmp_dict["shift_region"].replace("1/", ""))
    shift_pixel = int(crop_size/pieces)
    
    return int(base_width/shift_pixel) - (pieces-1)
    # -------------------------------------------------------------------------/