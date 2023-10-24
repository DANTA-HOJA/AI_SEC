import os
import sys
from typing import List, Dict, Tuple


"""
    (ref) github: https://github.com/jacobgil/pytorch-grad-cam
    
    vit_example: https://github.com/jacobgil/pytorch-grad-cam/blob/2183a9cbc1bd5fc1d8e134b4f3318c3b6db5671f/usage_examples/vit_example.py
"""

def reshape_transform(tensor, height=14, width=14):
    
    # Control how many sub-images are needed to be divided.
    #   When 'patch_size' is 16 and 'input_size' is 224,
    #   (14 * 14) sub-images will be generated.
    # 
    # Explaination: https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html#how-does-it-work-with-vision-transformers
    
    result = tensor[:, 1:, :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    
    return result