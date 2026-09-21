import numpy as np
from scipy import ndimage

def fill_holes_for_classes(mask:np.ndarray, target_classes=[3,4]) -> np.ndarray:
    """
    for multi-class semantic segmentation mask, 
    fill internal holes only for specified classes, keep other classes unchanged
    mask: H,W single-channel label array, int type
    target_classes: the class indices that need to fill holes, here is [3,4]
    return: new mask after processing
    """
    mask_out = mask.copy()
    for cls in target_classes:
        binary = (mask_out == cls)
        filled_binary = ndimage.binary_fill_holes(binary)
        hole_pixels = filled_binary & (~binary)
        mask_out[hole_pixels] = cls
    return mask_out


