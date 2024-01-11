import numpy as np
import matplotlib.pyplot as plt

"""Author:Kevin Paulose"""

def plot_flow(image, flow_image, confidence, threshmin):
    """
    Plot a flow field of one frame of the data.
    
    Inputs:
        - image: grayscale image - shape: (H, W)
        - flow_image: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - threshmin: threshold for confidence (optional) - scalar
    """
    
    ### STUDENT CODE START ###
    
    # Create a grid of x and y coordinates to represent the pixel locations
    H, W = image.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    
    # Select pixels where confidence is greater than the threshold
    mask = confidence > threshmin
    
    # Calculate the x and y components of the flow vectors
    u = flow_image[:, :, 0]
    v = flow_image[:, :, 1]
    
    # Plot the vector field
    plt.imshow(image, cmap='gray')
    plt.quiver(x[mask], y[mask], u[mask], v[mask], color='red', angles='xy', scale=55, width=0.0004)    
    ### STUDENT CODE END ###

    # this function has no return value

# image = np.random.rand(100, 100)  # Example image
# flow_image = np.random.rand(100, 100, 2)  # Example flow_image
# confidence = np.random.rand(100, 100)  # Example confidence

# plot_flow(image, flow_image, confidence, threshmin=0.5)  # Adjust the threshold as needed