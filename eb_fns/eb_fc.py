import numpy as np


def getMWPfc(top_MWP, top_weights, bottom_activations, contrast=False):
    # N_topMWP, H_topMWP, W_topMWP, C_topMWP = top_MWP.shape
    # H, W, C_in, C_out = top_weights.shape
    # N_bot_act, H_bot_act, W_bot_act, C_bot_act = bottom_activations.shape

    # H, W, C_in, C_out = top_weights.shape
    # top_weights_reshaped = top_weights.reshape(-1, C_out)
    # top_weights_clipped = top_weights_reshaped.clip(min=0)  # threshold weights at 0
    # m = np.dot(top_weights_clipped.T, bottom_activations)  # 1000 x 1
    # n = top_MWP / m  # 1000 x 1
    # o = np.dot(top_weights_clipped, n)  # 4096 x 1
    # bottom_MWP = (bottom_activations * o)  # 4096 x 1
    # # bottom_MWP = (bottom_activations * o).reshape(H, W, C_in)  # 4096 x 1
    # return bottom_MWP

    top_MWP_reshaped = top_MWP.reshape(-1, top_MWP.shape[-1])  # 1 x 1000 Reshape MWP as 1 x N row vector
    top_weights_reshaped = top_weights.reshape(-1, top_weights.shape[-1])  # 4096 x 1000
    # bottom_activations_reshaped = bottom_activations.reshape(-1, bottom_activations.shape[-1])  # 1 x 4096
    bottom_activations_reshaped = bottom_activations.reshape(1, -1)  # 1 x 4096

    top_weights_reshaped = top_weights_reshaped.clip(min=0)  # threshold weights at 0
    m = np.dot(bottom_activations_reshaped, top_weights_reshaped)  # 1 x 1000
    n = top_MWP_reshaped / m  # 1 x 1000
    o = np.dot(n, top_weights_reshaped.T)  # 1 x 4096
    bottom_MWP = (bottom_activations_reshaped * o)  # 1 x 4096
    bottom_MWP = bottom_MWP.reshape(bottom_activations.shape)   # 1 x 1 x 1 x 4096
    return bottom_MWP