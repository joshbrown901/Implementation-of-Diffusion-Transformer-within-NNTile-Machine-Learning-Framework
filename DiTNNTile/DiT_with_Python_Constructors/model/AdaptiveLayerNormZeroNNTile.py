import nntile
import numpy as np
nntile_config = nntile.starpu.Config(1,0,0,0)
nntile.starpu.init()
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, TensorTraits
import torch
import torch.nn as nn


from mlp_nntile import MLPNNTile
from scale_shift_nntile import ScaleShiftNNTile
from layer_norm import manual_layer_norm

class AdaptiveLayerNormZeroNNTile:
    def __init__(self, B, N, D, W0, W1, W2, W3, W4, W5, b0, b1, b2, b3, b4, b5, bias=True):
        super().__init__()
        self.B = B
        self.N = N
        self.D = D
        self.bias = bias
        
        self.W0 = W0
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.W4 = W4
        self.W5 = W5
        
        if self.bias:
            self.b0 = b0
            self.b1 = b1
            self.b2 = b2
            self.b3 = b3
            self.b4 = b4
            self.b5 = b5
        else:
            self.b0 = self.b1 = self.b2 = self.b3 = self.b4 = self.b5 = None


    def forward(self, x, emb):
        self.x = x
        self.emb = emb
        self.mlp_nntile = MLPNNTile(
        self.B, self.D,
        self.W0, self.W1, self.W2, self.W3, self.W4, self.W5,
        self.b0, self.b1, self.b2, self.b3, self.b4, self.b5,
        bias=True
    )
        self.shift_msa, self.scale_msa, self.gate_msa, self.shift_mlp, self.scale_mlp, self.gate_mlp = self.mlp_nntile.forward(self.emb)
        self.norm_x = nn.LayerNorm(self.D, elementwise_affine=False, eps=1e-6)
        self.norm_x_x = self.norm_x(self.x)
        self.nntile_scale_shift = ScaleShiftNNTile(self.B, self.N, self.D)
        self.x_out = self.nntile_scale_shift.forward(self.norm_x_x.detach().numpy(), self.scale_msa, self.shift_msa)
        return self.x_out, self.gate_msa, self.shift_mlp, self.scale_mlp, self.gate_mlp

    def backward(self, x, d_x_out, d_gate_msa, d_shift_mlp, d_scale_mlp, d_gate_mlp):
        self.x = x
        self.d_x_out = d_x_out
        self.d_gate_msa = d_gate_msa
        self.d_shift_mlp = d_shift_mlp
        self.d_scale_mlp = d_scale_mlp
        self.d_gate_mlp = d_gate_mlp
        self.d_norm_x_x, self.d_scale_msa, self.d_shift_msa = self.nntile_scale_shift.backward(self.d_x_out)        
        self.norm_x_x.backward(torch.tensor(self.d_norm_x_x), retain_graph=True)
        self.d_x = self.x.grad.clone().detach().numpy()
        self.x.grad.zero_()
        self.d_emb, self.dW0, self.dW1, self.dW2, self.dW3, self.dW4, self.dW5, self.db0, self.db1, self.db2, self.db3, self.db4, self.db5 = self.mlp_nntile.backward(self.emb, self.d_shift_msa, self.d_scale_msa, self.d_gate_msa, self.d_shift_mlp, self.d_scale_mlp, self.d_gate_mlp)
        return self.d_x, self.d_emb