import numpy as np

"""Author:Kevin Paulose"""

def epipole(flow_x, flow_y, smin, thresh, num_iterations=None):
    """
    Compute the epipole from the flows,

    Inputs:
        - flow_x: optical flow on the x-direction - shape: (H, W)
        - flow_y: optical flow on the y-direction - shape: (H, W)
        - smin: confidence of the flow estimates - shape: (H, W)
        - thresh: threshold for confidence - scalar
        - Ignore num_iterations
    Outputs:
        - ep: epipole - shape: (3,)
    """
    # Logic to compute the points you should use for your estimation
    # We only look at image points above the threshold in our image
    # Due to memory constraints, we cannot use all points on the autograder
    # Hence, we give you valid_idx which are the flattened indices of points
    # to use in the estimation estimation problem
    good_idx = np.flatnonzero(smin > thresh)
    permuted_indices = np.random.RandomState(seed=10).permutation(good_idx)
    valid_idx = permuted_indices[:3000]

    # Extract the coordinates of the valid points
    x = valid_idx % flow_x.shape[1]
    y = valid_idx // flow_x.shape[1]

    # Extract the optical flow vectors at the valid points
    u = flow_x[y, x]
    v = flow_y[y, x]
    x=x-flow_x.shape[1]//2
    y=y-flow_y.shape[1]//2
    # Create a matrix of homogeneous coordinates for the valid points
    points = np.column_stack((x, y, np.ones(len(x))))

    # Create a matrix of optical flow vectors for the valid points
    flow_vectors = np.column_stack((u, v,np.zeros(len(x))))

    # Solve the linear system A * ep = 0 for the epipole
    A = np.cross(points, flow_vectors)
    _, _, V = np.linalg.svd(A,full_matrices=False)
    ep = V[-1]

    return ep