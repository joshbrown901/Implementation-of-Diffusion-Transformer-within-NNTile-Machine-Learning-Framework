import nntile
import numpy as np
nntile_config = nntile.starpu.Config(1,0,0,0)
nntile.starpu.init()
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, TensorTraits
from typing import List, Union
import torch  
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

def relative_error(a, b, eps=1e-8):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + eps)

# ---------------------------
# NumPy implementation
# ---------------------------
class AdaptiveLayerNumpy:
    def __init__(self, H):
        self.H = H
        # Shared activation → then two separate linears
        self.W1 = np.random.randn(H, H).astype(np.float32)
        self.b1 = np.random.randn(H).astype(np.float32)
        self.W2 = np.random.randn(H, H).astype(np.float32)
        self.b2 = np.random.randn(H).astype(np.float32)

    @staticmethod
    def silu(x):
        sig = 1 / (1 + np.exp(-x))
        return x * sig

    @staticmethod
    def dsilu(x):
        sig = 1 / (1 + np.exp(-x))
        return sig + x * sig * (1 - sig)

    def forward(self, x):
        self.x = x                                   # (B, H)
        self.a = self.silu(x)                        # (B, H)
        self.y1 = self.a @ self.W1.T + self.b1       # (B, H)
        self.y2 = self.a @ self.W2.T + self.b2       # (B, H)
        return self.y1, self.y2                      # (B, H), (B, H)

    def backward(self, grad_y1, grad_y2):
        # Gradient of outputs w.r.t. activation
        grad_a1 = grad_y1 @ self.W1                  # (B, H)
        grad_a2 = grad_y2 @ self.W2                  # (B, H)
        grad_a  = grad_a1 + grad_a2                  # combine gradients

        # Backprop through SiLU
        grad_x = grad_a * self.dsilu(self.x)         # (B, H)

        # Weight/bias grads
        grad_W1 = grad_y1.T @ self.a                 # (H, H)
        grad_b1 = grad_y1.sum(axis=0)                # (H,)
        grad_W2 = grad_y2.T @ self.a                 # (H, H)
        grad_b2 = grad_y2.sum(axis=0)                # (H,)

        return grad_x, grad_W1, grad_b1, grad_W2, grad_b2

# ---------------------------
# PyTorch implementation
# ---------------------------
class AdaptiveLayerTorch(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.act = nn.SiLU()
        self.fc1 = nn.Linear(H, H)
        self.fc2 = nn.Linear(H, H)

    def forward(self, x):
        a = self.act(x)
        y1 = self.fc1(a)
        y2 = self.fc2(a)
        return y1, y2
