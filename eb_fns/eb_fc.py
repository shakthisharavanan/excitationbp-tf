import numpy as np


def getMWPfc(top_MWP, top_weights, bottom_activations, contrast=False):
    H, W, C_in, C_out = top_weights.shape
    top_weights_reshaped = top_weights.reshape(-1, C_out)
    top_weights_clipped = top_weights_reshaped.clip(min=0)  # threshold weights at 0
    m = np.dot(top_weights_clipped.T, bottom_activations)  # 1000 x 1
    n = top_MWP / m  # 1000 x 1
    o = np.dot(top_weights_clipped, n)  # 4096 x 1
    bottom_MWP = (bottom_activations * o)  # 4096 x 1
    # bottom_MWP = (bottom_activations * o).reshape(H, W, C_in)  # 4096 x 1
    return bottom_MWP
