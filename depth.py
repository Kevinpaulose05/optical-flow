import numpy as np

"""Author:Kevin Paulose"""

def depth(flow, confidence, ep, K, thres=10):
    """
    Compute the depth map from the flow and confidence map.

    Inputs:
        - flow: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - ep: epipole - shape: (3,)
        - K: intrinsic calibration matrix - shape: (3, 3)
        - thres: threshold for confidence (optional) - scalar

    Output:
        - depth_map: depth at every pixel - shape: (H, W)
    """
    depth_map = np.zeros_like(confidence)

    ### STUDENT CODE START ###

    # 1. Find where flow is valid (confidence > threshold)
    valid_flow_indices = confidence > thres

    # 2. Convert these pixel locations to normalized projective coordinates
    valid_pixel_indices = np.column_stack(np.where(valid_flow_indices))
    valid_pixel_coordinates = np.column_stack((valid_pixel_indices, np.ones(valid_pixel_indices.shape[0])))
    # print(valid_pixel_coordinates[0:2])
    
    # 3. Normalize epipole and flow vectors
    ep = ep.reshape(3,1)
    ep_normalized = np.linalg.inv(K) @ ep
    valid_pixel_coordinates = np.linalg.inv(K) @ valid_pixel_coordinates.T
    flow_normalized = flow[valid_flow_indices]
    # print(valid_pixel_indices.shape)
    
    # 4. Find the depths using the formula from the lecture slides
    shifted_points = valid_pixel_coordinates - ep_normalized
    # print( np.linalg.norm(flow_normalized,axis=1).shape, np.linalg.norm(shifted_points,axis=0).shape)
    depth =  np.linalg.norm(shifted_points,axis=0) / np.linalg.norm(flow_normalized,axis=1)

    # Fill in the depth map with the computed depths
    depth_map[valid_flow_indices] = depth

    # Require depths to be positive
    truncated_depth_map = np.maximum(depth_map, 0)
    valid_depths = truncated_depth_map[truncated_depth_map > 0]

    # Depth bound for better visualisation
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)

    # Set depths above the bound to 0 and normalize to [0, 1]
    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()

    return truncated_depth_map