import os
import numpy as np
from skimage import morphology, measure
from skimage.draw import rectangle


def dump_info(image, CLI_print_title:str=None):
    """ A handy function to print details of an image object.
    """
    name = image.name if hasattr(image, 'name') else None # xarray
    if name is None and hasattr(image, 'getName'): name = image.getName() # Dataset
    if name is None and hasattr(image, 'getTitle'): name = image.getTitle() # ImagePlus
    
    if CLI_print_title is not None: print(f"--> {CLI_print_title}:\n")
    print(f"    name  : {name or 'N/A'}")
    print(f"    type  : {type(image)}")
    print(f"    dtype : {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
    print(f"    shape : {image.shape if hasattr(image, 'shape') else 'N/A'}")
    print(f"    dims  : {image.dims if hasattr(image, 'dims') else 'N/A'}\n")



def group_show(*args):
    """
    """
    for image in args: image.show()



def group_hide(*args):
    """
    """
    for image in args: image.hide()



# def save_tif_with_SN(img, save_name, folder_path, tif_save_cnt, tif_save_SNdigits, ij):
#     """
#     """
#     ij.IJ.saveAsTiff(img, os.path.normpath(f"{folder_path}/{tif_save_cnt:{tif_save_SNdigits}}_{save_name}.tif"))
#     tif_save_cnt += 1

#     return tif_save_cnt



# def direct_max_zproj(img_list, ZProjector, RGBStackMerge, RGBStackConverter):
#     """
#     """
#     ch_B = img_list[0]
#     ch_G = img_list[1]
#     ch_R = img_list[2]
    
#     max_zproj_ch_B = ZProjector.run(ch_B, "max")
#     max_zproj_ch_G = ZProjector.run(ch_G, "max")
#     max_zproj_ch_R = ZProjector.run(ch_R, "max")
    
#     stack_RGB = RGBStackMerge.mergeChannels([max_zproj_ch_R, max_zproj_ch_G, max_zproj_ch_B], True)
    
#     RGBStackConverter.convertToRGB(stack_RGB)
    
#     return stack_RGB



# def median_R1_and_mean3D_R2(image, ZProjector, ij):
#     """
#     """
#     median_r1 = image.duplicate()
#     ij.IJ.run(median_r1, "Median...", "radius=1 stack")
    
#     median_r1_mean3D_r2 = median_r1.duplicate()
#     ij.IJ.run(median_r1_mean3D_r2, "Mean 3D...", "x=2 y=2 z=2")
    
#     # ij.IJ.run(median_r1_mean3D_r2, "Z Project...", "projection=[Max Intensity]")
#     # median_r1_mean3D_r2_Zproj_max = ij.WindowManager.getCurrentImage()
#     # median_r1_mean3D_r2_Zproj_max.hide()
#     median_r1_mean3D_r2_Zproj_max = ZProjector.run(median_r1_mean3D_r2, "max") # 'method' is "avg", "min", "max", "sum", "sd" or "median".
    
#     return median_r1_mean3D_r2_Zproj_max



# def crop_threshold_rect(image, roi, ij):
#     """
#     """
#     image.setRoi(roi)
#     ij.IJ.run(image, "Fit Rectangle", "")
#     cropped_img = image.crop()
    
#     return cropped_img



# def create_rect(img: np.ndarray, relative_pos:str) -> np.ndarray:
#     """
#     """
#     if (relative_pos!= "inner") and (relative_pos!="outer"): raise ValueError("relative_pos can only be 'inner' or 'outer'. ")
    
#     contours = measure.find_contours(img)
#     # print(len(contours))
    
#     # Get all contours' coordinate
#     all_coord = []
#     for contour in contours:
#         # print(len(contour))
#         for pt in contour:
#             all_coord.append(pt)
#             # print(pt)
#         # print(len(all_coord))
    
#     # Find pt of outer rectangle
#     max_Y = img.shape[1]/2
#     min_Y = img.shape[1]/2
#     max_X = img.shape[0]/2
#     min_X = img.shape[0]/2
#     for pt in all_coord:
#         if pt[0] > max_Y: max_Y = int(pt[0])
#         if pt[0] < min_Y: min_Y = int(pt[0])
#         if pt[1] > max_X: max_X = int(pt[1])
#         if pt[1] < min_X: min_X = int(pt[1])    
#     # print(f"outer --> min_X, max_X, min_Y, max_Y = {min_X}, {max_X}, {min_Y}, {max_Y}")
    
#     # Find pt of inner rectangle
#     if relative_pos == "inner":
#         middle_X = (max_X + min_X)/2
#         bone_L = middle_X-50
#         bone_R = middle_X+50
#         max_X = img.shape[0]
#         min_X = 0
#         for pt in all_coord:
#             if (pt[1] > bone_R) and (pt[1] < max_X): max_X = int(pt[1])
#             if (pt[1] < bone_L) and (pt[1] > min_X): min_X = int(pt[1])
#     # print(f"inner --> min_X, max_X = {min_X}, {max_X}")
    
#     # Make rectangle image
#     rr, cc = rectangle(start=(0, min_X), end=(1024, max_X), shape=img.shape)
#     # print(rr, cc)
#     rect = np.zeros_like(img)
#     rect[rr, cc] = 255
    
#     return rect