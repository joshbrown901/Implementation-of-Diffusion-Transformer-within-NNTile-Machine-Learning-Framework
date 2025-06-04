import torch
import torch.nn as nn

class ScalePlusSkipConnectionTorch(nn.Module):
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # scale, shift are (B,D); avoid any in-place ops
        scale_mod = scale + 1.0                   # out-of-place
        y = x * scale_mod.unsqueeze(1) + shift
        return y