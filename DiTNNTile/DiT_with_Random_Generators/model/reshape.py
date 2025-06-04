import numpy as np
import torch

# --- NumPy implementation ---
def unpatchify_numpy(x, nh, nw, ph, pw, im_channels):
    B, num_patches, patch_dim = x.shape
    assert num_patches == nh * nw, f"Expected {nh*nw} patches, got {num_patches}"
    # reshape into (B, nh, nw, ph, pw, C)
    x = x.reshape(B, nh, nw, ph, pw, im_channels)
    # transpose to (B, C, nh, ph, nw, pw)
    x = np.transpose(x, (0, 5, 1, 3, 2, 4))
    # collapse to (B, C, nh*ph, nw*pw)
    return x.reshape(B, im_channels, nh * ph, nw * pw)


# --- PyTorch implementation ---
def unpatchify_torch(x, nh, nw, ph, pw, im_channels):
    B, num_patches, patch_dim = x.shape
    assert num_patches == nh * nw, f"Expected {nh*nw} patches, got {num_patches}"
    x = x.reshape(B, nh, nw, ph, pw, im_channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, im_channels, nh * ph, nw * pw)