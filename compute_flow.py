import numpy as np

"""Author:Kevin Paulose"""

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    Find the Lucas-Kanade optical flow on a single square patch.
    The patch is centered at (y, x), therefore it generally extends
    from x-size//2 to x+size//2 (inclusive), same for y, EXCEPT when
    exceeding image boundaries!
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative - shape: (H, W)
        - x: SECOND coordinate of patch center - integer in range [0, W-1]
        - y: FIRST coordinate of patch center - integer in range [0, H-1]
        - size: optional parameter to change the side length of the patch in pixels
    
    Outputs:
        - flow: flow estimate for this patch - shape: (2,)
        - conf: confidence of the flow estimates - scalar
    """

    ### STUDENT CODE START ###
    half_size = size // 2

    # Define the patch boundaries
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size, Ix.shape[1] - 1)
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size, Ix.shape[0] - 1)

    # Extract the patch gradients
    Ix_patch = Ix[y_min:y_max+1, x_min:x_max+1]
    Iy_patch = Iy[y_min:y_max+1, x_min:x_max+1]
    It_patch = It[y_min:y_max+1, x_min:x_max+1]

    # Reshape the patch gradients into 1D arrays
    Ix_patch = Ix_patch.flatten()
    Iy_patch = Iy_patch.flatten()
    It_patch = -It_patch.flatten()  # Negate It as in Ixu + Iyv + It = 0

    # Construct the coefficient matrix A
    A = np.vstack((Ix_patch, Iy_patch)).T

    # Solve the linear system using least squares (set rcond=-1 for accurate results)
    flow, _, _, _ = np.linalg.lstsq(A, It_patch, rcond=-1)
    
    # Calculate the confidence as the smallest singular value of A
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    smin = np.min(s)
    
    ### STUDENT CODE END ###

    return flow, smin

def flow_lk(Ix, Iy, It, size=5):
    """
    Compute the Lucas-Kanade flow for all patches of an image.
    To do this, iteratively call flow_lk_patch for all possible patches.
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative - shape: (H, W)
    Outputs:
        - image_flow: flow estimate for each patch - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
    """

    H, W = Ix.shape
    image_flow = np.zeros((H, W, 2))
    confidence = np.zeros((H, W))

    for y in range(0, H ):
        for x in range(0, W):
            flow, smin = flow_lk_patch(Ix, Iy, It, x, y, size)
            image_flow[y, x, 0] = flow[0]  # Assign u component
            image_flow[y, x, 1] = flow[1]  # Assign v component
            confidence[y, x] = smin

    return image_flow, confidence