import numpy as np
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = config['head_dim']
        self.att_dim = self.n_heads * self.head_dim

        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.att_dim, bias=True)
        self.output_proj = nn.Linear(self.att_dim, self.hidden_size, bias=True)

    def forward(self, x):
        B, N = x.shape[:2]
        q, k, v = self.qkv_proj(x).split(self.att_dim, dim=-1)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).reshape(B, N, self.att_dim)
        out = self.output_proj(out)
        return out