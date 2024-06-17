import cv2
import numpy as np
# -----------------------------------------------------------------------------/


def create_brightness_mask(rgb_img: np.ndarray,
                        br_thres: int,
                        erode_kernel: np.ndarray=None, erode_iter:int=1,
                        dilate_kernel: np.ndarray=None, dilate_iter:int=1,
                        ) ->  tuple[np.ndarray, float]:
    """ workflow:\n\t rgb_img -> hsv_img -> br_thres mask \
        -> erode (optional) -> dilate (optional)

    Args:
        rgb_img (np.ndarray): image in RGB order, 8-bit, 3-channel
        br_thres (int): a value apply threshold to V(B) channel of HSV(HSB)
        erode_kernel (np.ndarray, optional): Defaults to None.
        erode_iter (int, optional): erode iterations. Defaults to 1.
        dilate_kernel (np.ndarray, optional): Defaults to None.
        dilate_iter (int, optional): dilate iterations. Defaults to 1.

    Returns:
        tuple[np.ndarray, float]: (`br_mask`, `dark_ratio`)
    """
    # convert RGB to HSV(HSB), and get V(B) channel
    brightness = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV_FULL)[:,:,2]
    mask = brightness > br_thres
    
    # erode
    if erode_kernel is not None:
        tmp = cv2.erode(np.uint8(mask*255),
                        erode_kernel, iterations=erode_iter)
        mask = np.bool8(tmp)
    
    # dilate
    if dilate_kernel is not None:
        tmp = cv2.dilate(np.uint8(mask*255),
                            dilate_kernel, iterations=dilate_iter)
        mask = np.bool8(tmp)
    
    # dark_ratio
    pixel_too_dark = np.sum(~mask)
    dark_ratio = pixel_too_dark/(rgb_img.shape[0]*rgb_img.shape[1])
    
    return mask, dark_ratio
    # -------------------------------------------------------------------------/


def thres_cam_on_cell_v1(cam_img: np.ndarray,
                         cam_thres: int,
                         br_mask: np.ndarray,
                         erode_kernel: np.ndarray=None, erode_iter:int=1,
                         dilate_kernel: np.ndarray=None, dilate_iter:int=1,
                         ) -> np.ndarray:
    """ workflow:\n\t thres(cam) -> thres(cam)*mask \
        -> erode (optional) -> dilate (optional)

    Args:
        cam_img (np.ndarray): cam image (grayscale), 8-bit, 1-channel
        cam_thres (int): a value apply threshold to cam image
        br_mask (np.ndarray): br_mask generate by `create_brightness_mask()`
        erode_kernel (np.ndarray, optional): Defaults to None.
        erode_iter (int, optional): erode iterations. Defaults to 1.
        dilate_kernel (np.ndarray, optional): Defaults to None.
        dilate_iter (int, optional): dilate iterations. Defaults to 1.

    Returns:
        np.ndarray: `thresholded_cam_on_cell`
    """
    assert br_mask.dtype == np.bool8, "`br_mask` dtype should be `np.bool`"
    
    # thres(cam)
    _, thres_cam = cv2.threshold(cam_img,
                                 cam_thres, 255,
                                 cv2.THRESH_BINARY)
    
    # thres(cam)*mask
    masked_thres_cam = thres_cam * br_mask
    
    # erode
    if erode_kernel is not None:
        masked_thres_cam = cv2.erode(masked_thres_cam,
                                     erode_kernel, iterations=erode_iter)
    
    # dilate
    if dilate_kernel is not None:
        tmp = cv2.dilate(masked_thres_cam,
                            dilate_kernel, iterations=dilate_iter)
    
    return masked_thres_cam
    # -------------------------------------------------------------------------/


def thres_cam_on_cell_v2(cam_img: np.ndarray,
                         cam_thres: int,
                         br_mask: np.ndarray,
                         erode_kernel: np.ndarray=None, erode_iter:int=1,
                         dilate_kernel: np.ndarray=None, dilate_iter:int=1,
                         ) -> np.ndarray:
    """ workflow:\n\t cam*mask -> thres(cam*mask) \
        -> erode (optional) -> dilate (optional)

    Args:
        cam_img (np.ndarray): cam image (grayscale), 8-bit, 1-channel
        cam_thres (int): a value apply threshold to cam image
        br_mask (np.ndarray): br_mask generate by `create_brightness_mask()`
        erode_kernel (np.ndarray, optional): Defaults to None.
        erode_iter (int, optional): erode iterations. Defaults to 1.
        dilate_kernel (np.ndarray, optional): Defaults to None.
        dilate_iter (int, optional): dilate iterations. Defaults to 1.

    Returns:
        np.ndarray: `thresholded_cam_on_cell`
    """
    assert br_mask.dtype == np.bool8, "`br_mask` dtype should be `np.bool`"
    
    # cam*mask
    masked_cam = cam_img * br_mask
    
    # thres(cam*mask)
    _, thres_masked_cam = cv2.threshold(masked_cam,
                                        cam_thres, 255,
                                        cv2.THRESH_BINARY)
    
    # erode
    if erode_kernel is not None:
        thres_masked_cam = cv2.erode(thres_masked_cam,
                                     erode_kernel, iterations=erode_iter)
    
    # dilate
    if dilate_kernel is not None:
        tmp = cv2.dilate(thres_masked_cam,
                            dilate_kernel, iterations=dilate_iter)
    
    return thres_masked_cam
    # -------------------------------------------------------------------------/


def calc_area(binary_img:np.ndarray) -> list[float]:
    """
    """
    # find contours
    # contours is a list of regions, each region contains ponits to form a closed area.
    contours, _ = \
        cv2.findContours(binary_img,  cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # count area
    contour_areas = []
    for contour in contours:
        area = cv2.contourArea(contour) # 取得填充區域面積
        if (area > 1):
            contour_areas.append(area)
            
    return contour_areas
    # -------------------------------------------------------------------------/


def calc_thresed_cam_area_on_cell(orig_img:np.ndarray,
                                  br_thres: int,
                                  cam_img:np.ndarray,
                                  cam_thres: int) -> list[float]:
    """ create_brightness_mask() -> cam_thres_on_cell_v2() -> calc_area()
    """
    img_dict: dict[str, np.ndarray] = {}
    img_dict["kernel_ones2x2"] = np.ones((2, 2), dtype=np.uint8)
    img_dict["kernel_ones3x3"] = np.ones((3, 3), dtype=np.uint8)
    
    # create_brightness_mask()
    br_mask, _ = create_brightness_mask(orig_img, br_thres,
                                        erode_kernel=img_dict["kernel_ones3x3"],
                                        erode_iter=1,
                                        dilate_kernel=img_dict["kernel_ones2x2"],
                                        dilate_iter=1)
    
    # cam_thres_on_cell_v2()
    cam_thres_on_cell = \
        thres_cam_on_cell_v2(cam_img, cam_thres, br_mask,
                                erode_kernel=img_dict["kernel_ones2x2"],
                                erode_iter=1)
    
    # ones = np.ones_like(cam_img, dtype=np.bool8)
    # cam_thres_on_cell = \
    #     cam_thres_on_cell_v2(cam_img, cam_thres,
    #                             ones, erode_kernel=(3, 3))
    
    # calc_area()
    contour_areas = calc_area(cam_thres_on_cell)
            
    return contour_areas
    # -------------------------------------------------------------------------/