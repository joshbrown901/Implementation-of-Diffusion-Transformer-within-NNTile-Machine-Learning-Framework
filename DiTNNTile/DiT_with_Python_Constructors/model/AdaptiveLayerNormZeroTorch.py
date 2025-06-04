# Adaptive Layer Normalization Pytorch (Main Implementation)
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLayerNormZeroTorch(nn.Module):
    def __init__(self, embedding_dim, bias=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bias_flag = bias
        self.W = nn.Parameter(torch.randn(6 * embedding_dim, embedding_dim))
        self.b = nn.Parameter(torch.randn(6 * embedding_dim)) if bias else None
        self.eps = 1e-6

    def linear(self, x):
        out = F.linear(x, self.W, self.b)
        return out

    def forward(self, x, emb):
        activated_emb = F.silu(emb)
        lin_out = self.linear(activated_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(lin_out, 6, dim=-1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        x_out = norm_x * (1 + scale_msa[:, None]) + shift_msa[:, None]

        return x_out, gate_msa, shift_mlp, scale_mlp, gate_mlp