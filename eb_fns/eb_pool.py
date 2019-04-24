import numpy as np


def getMWPmaxpool(top_MWP, bottom_activations, gradient):
    N_topMWP, H_topMWP, W_topMWP, C_topMWP = top_MWP.shape  # find shapes of pool input and output
    N_bot_act, H_bot_act, W_bot_act, C_bot_act = bottom_activations.shape

    # Calculate number of repeats that need to happen to enlarge pool output volume size to pool input volume size
    H_repeat = int(H_bot_act / H_topMWP)
    W_repeat = int(W_bot_act / W_topMWP)

    top_MWP_repeated = top_MWP.repeat(H_repeat, axis=1).repeat(W_repeat, axis=2)  # (1, 14, 14, 512)

    # Multiply repeated top_MWP to gradients (binary mapping) to get corressponding
    bottom_MWP = gradient * top_MWP_repeated
    return bottom_MWP