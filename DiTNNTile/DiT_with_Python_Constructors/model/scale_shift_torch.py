import torch
import torch.nn as nn


# PyTorch ScaleShift class
class ScaleShiftTorch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, scale, shift):
           # scale, shift are (B,D)
        scale_mod = scale.clone().add_(1.0)      # (B,D)
        # unsqueeze to broadcast along N
        y = x * scale_mod.unsqueeze(1) + shift.unsqueeze(1)
        return y