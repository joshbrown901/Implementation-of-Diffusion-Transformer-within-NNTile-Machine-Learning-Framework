import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class MLPTorch(nn.Module):
    def __init__(self, embedding_dim, bias=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.randn(6 * embedding_dim, embedding_dim))
        self.b = nn.Parameter(torch.randn(6 * embedding_dim)) if bias else None

    def forward(self, emb):
        activated = F.silu(emb)                            
        lin = F.linear(activated, self.W, self.b)        
        return torch.chunk(lin, 6, dim=-1)