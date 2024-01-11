import numpy as np
from scipy.ndimage import convolve1d

"""Author:Kevin Paulose"""

# Define the kernels for computing image gradients
# KERNEL_G is a Gaussian smoothing kernel for smoothing across orthogonal axes after gradient computation
# KERNEL_H is a derivative kernel that measures the difference between neighboring pixels (derivative of a Gaussian)
KERNEL_G = np.array([0.015625, 0.093750, 0.234375, 0.312500, 0.234375, 0.093750, 0.015625])
KERNEL_H = np.array([0.03125, 0.12500, 0.15625, 0, -0.15625, -0.1250, -0.03125])

def compute_Ix(imgs):
    """
    Compute the gradient of the images along the x-dimension.
    
    WARNING: The first coordinate of the gradient is actually the y-coordinate,
    which means you're computing the gradient along axis=1, NOT 0!
    
    Inputs:
        - imgs: image volume, where the first dimension represents y, the second represents x, and the third represents time (t).
        - Shape: (H, W, N) where H is the height, W is the width, and N is the number of frames.
    Outputs:
        - Ix: Image gradient along the x-dimension.
        - Shape: (H, W, N)
    """
        
    # Compute the derivative along the time dimension for each frame
    Ix = convolve1d(imgs, KERNEL_H, axis=1)
    
    # Smooth the derivative along the x-dimension using KERNEL_G
    Ix = convolve1d(Ix, KERNEL_G, axis=0)
    Ix = convolve1d(Ix, KERNEL_G, axis=2)
    
    return Ix

def compute_Iy(imgs):
    """
    Compute the gradient of the images along the y-dimension.
    
    WARNING: The first coordinate of the gradient is actually the y-coordinate,
    which means you're computing the gradient along axis=0, NOT 1!
    
    Inputs:
        - imgs: image volume, where the first dimension represents y, the second represents x, and the third represents time (t).
        - Shape: (H, W, N) where H is the height, W is the width, and N is the number of frames.
    Outputs:
        - Iy: Image gradient along the y-dimension.
        - Shape: (H, W, N)
    """
    
    # Compute the derivative along the time dimension for each frame
    Iy = convolve1d(imgs, KERNEL_H, axis=0)
    
    # Smooth the derivative along the x-dimension using KERNEL_G
    Iy = convolve1d(Iy, KERNEL_G, axis=1)
    Iy = convolve1d(Iy, KERNEL_G, axis=2)
    
    return Iy

def compute_It(imgs):
    """
    Compute the gradient of the images along the t-dimension.
    
    WARNING: The first coordinate of the gradient is actually the y-coordinate,
    which means you're computing the gradient along axis=0, NOT 1!
    
    Inputs:
        - imgs: image volume, where the first dimension represents y, the second represents x, and the third represents time (t).
        - Shape: (H, W, N) where H is the height, W is the width, and N is the number of frames.
    Outputs:
        - It: Temporal image gradient representing changes over time.
        - Shape: (H, W, N)
    """
    
    # Compute the derivative along the time dimension for each frame
    It = convolve1d(imgs, KERNEL_H, axis=2)
    
    # Smooth the derivative along the x-dimension using KERNEL_G
    It = convolve1d(It, KERNEL_G, axis=0)
    It = convolve1d(It, KERNEL_G, axis=1)
    
    return It
