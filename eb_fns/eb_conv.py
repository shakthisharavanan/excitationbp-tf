import numpy as np
from im2col import *


def getMWPconv(top_MWP, top_weights, bottom_activations):
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

    # top_MWP_reshaped = top_MWP.reshape(-1, top_MWP.shape[-1])  # 1 x 1000 Reshape MWP as 1 x N row vector
    # top_weights_reshaped = top_weights.reshape(-1, top_weights.shape[-1])  # 4096 x 1000
    # # bottom_activations_reshaped = bottom_activations.reshape(-1, bottom_activations.shape[-1])  # 1 x 4096
    # bottom_activations_reshaped = bottom_activations.reshape(1, -1)  # 1 x 4096
    #
    # top_weights_reshaped = top_weights_reshaped.clip(min=0)  # threshold weights at 0
    # m = np.dot(bottom_activations_reshaped, top_weights_reshaped)  # 1 x 1000
    # n = top_MWP_reshaped / m  # 1 x 1000
    # o = np.dot(n, top_weights_reshaped.T)  # 1 x 4096
    # bottom_MWP = (bottom_activations_reshaped * o)  # 1 x 4096
    # bottom_MWP = bottom_MWP.reshape(bottom_activations.shape)   # 1 x 1 x 1 x 4096

    """ For conv5_2 MWP """

    top_MWP_reshaped = top_MWP.reshape(-1, top_MWP.shape[-1])  # 196 x 512 Reshape MWP as 1 x N row vector
    top_weights_reshaped = top_weights.reshape(-1, top_weights.shape[-1])  # 4608 x 512

    # Reshape bottom activations in columns format
    bottom_activations_T = np.transpose(bottom_activations, [0, 3, 1, 2])
    bottom_activations_reshaped = im2col_indices(bottom_activations_T, 3, 3).T  # 4608 x 196

    # Calculate MWP of pool5 using Eq 10 in paper
    top_weights_reshaped = top_weights_reshaped.clip(min=0)  # threshold weights at 0
    m = np.dot(bottom_activations_reshaped, top_weights_reshaped)  # 196 x 512
    n = top_MWP_reshaped / m  # 196 x 512
    o = np.dot(n, top_weights_reshaped.T)  # 196 x 4608
    bottom_MWP = (bottom_activations_reshaped * o)  # 196 x 4608
    bottom_MWP = col2im_indices(bottom_MWP.T, bottom_activations_T.shape, 3, 3) # 1 x 512 x 14 x 14 (Transposed because col2im expects activations in column format)
    bottom_MWP = bottom_MWP.transpose([0, 2, 3, 1]) # 1 x 14 x 14 x 512

    return bottom_MWP