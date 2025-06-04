import numpy as np
import torch
import torch.nn as nn
from reshape import unpatchify_numpy, unpatchify_torch



def rel_error(a, b, eps=1e-8):
    abs_diff = np.abs(a - b)
    rel_err = abs_diff / (np.abs(b) + eps)
    max_rel = np.max(rel_err)
    mean_rel = np.mean(rel_err)
    return max_rel, mean_rel


if __name__ == "__main__":
    B, nh, nw, ph, pw, C = 2, 4, 3, 5, 6, 3
    num_patches = nh * nw
    patch_dim = ph * pw * C

    np.random.seed(42)
    x_np = np.random.randn(B, num_patches, patch_dim).astype(np.float32)
    x_torch = torch.tensor(x_np)

    out_np = unpatchify_numpy(x_np, nh, nw, ph, pw, C)
    out_torch = unpatchify_torch(x_torch, nh, nw, ph, pw, C).numpy()

    max_rel, mean_rel = rel_error(out_np, out_torch)
    print("max_rel", max_rel)
    print("mean_rel", mean_rel)