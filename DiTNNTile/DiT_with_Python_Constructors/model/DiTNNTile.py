from mlp_nntile import MLPNNTile
from layer_norm import manual_layer_norm
from scale_shift_nntile import ScaleShiftNNTile
from AttentionNumpy import Attention_numpy
from scale_plus_skipconnection_nntile import ScalePlusSkipConnectionNNTile
from PointwiseFeedForwardNNTile import FeedForwardNNTile
from TransformerLayerTorch import  TransformerLayer


import nntile
import numpy as np
import torch
import torch.nn as nn


class TransformerLayerNNTile:
    def __init__(self, B, N, F, D, num_heads, head_dim, W0, W1, W2, W3, W4, W5, 
                 b0, b1, b2, b3, b4, b5, W1_ff, b1_ff, W2_ff, b2_ff, 
                 qkv_weight, qkv_bias, out_weight, out_bias, bias=True):
        self.B = B
        self.N = N
        self.F = F
        self.D = D
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bias = bias

        # MLP weights
        self.W0 = nntc.from_array(W0)
        self.W1 = nntc.from_array(W1)
        self.W2 = nntc.from_array(W2)
        self.W3 = nntc.from_array(W3)
        self.W4 = nntc.from_array(W4)
        self.W5 = nntc.from_array(W5)
        
        if self.bias:
            self.b0 = nntc.from_array(b0)
            self.b1 = nntc.from_array(b1)
            self.b2 = nntc.from_array(b2)
            self.b3 = nntc.from_array(b3)
            self.b4 = nntc.from_array(b4)
            self.b5 = nntc.from_array(b5)
        else:
            self.b0 = self.b1 = self.b2 = self.b3 = self.b4 = self.b5 = None

        # Feed-forward weights
        self.W1_ff = nntc.from_array(W1_ff)
        self.b1_ff = nntc.from_array(b1_ff)
        self.W2_ff = nntc.from_array(W2_ff)
        self.b2_ff = nntc.from_array(b2_ff)

        # Attention weights
        self.qkv_weight = nntc.from_array(qkv_weight)
        self.qkv_bias = nntc.from_array(qkv_bias)
        self.out_weight = nntc.from_array(out_weight)
        self.out_bias = nntc.from_array(out_bias)
        

    def forward(self, x, emb):
        self.x = x  # Patched-x Torch Tensor
        self.emb = emb  # Numpy Tensor
        self.mlp_nntile = MLPNNTile(
            self.B, self.D,
            self.W0, self.W1, self.W2, self.W3, self.W4, self.W5,
            self.b0, self.b1, self.b2, self.b3, self.b4, self.b5,
            bias=self.bias
        )
        self.shift_msa, self.scale_msa, self.gate_msa, self.shift_mlp, self.scale_mlp, self.gate_mlp = self.mlp_nntile.forward(self.emb)
        
        self.norm_x = nn.LayerNorm(self.D, elementwise_affine=False, eps=1e-6)
        self.norm_x_x = self.norm_x(torch.tensor(x, dtype=torch.float32))  # Convert to torch tensor
        self.nntile_pre_attn_scale_shift = ScaleShiftNNTile(self.B, self.N, self.D)
        self.x_out_pre_attn = self.nntile_pre_attn_scale_shift.forward(
            self.norm_x_x.detach().numpy(), self.scale_msa, self.shift_msa)
        
        self.attn_np = Attention_numpy({
            'num_heads': self.num_heads,
            'hidden_size': self.D,
            'head_dim': self.head_dim
        })
        # Set Attention_numpy weights
        self.attn_np.qkv_weight = self.qkv_weight.get_array().copy()
        self.attn_np.qkv_bias = self.qkv_bias.get_array().copy()
        self.attn_np.out_weight = self.out_weight.get_array().copy()
        self.attn_np.out_bias = self.out_bias.get_array().copy()
        
        self.out_post_attn = self.attn_np.forward(self.x_out_pre_attn)
        self.nntile_post_attn_scale_skip = ScalePlusSkipConnectionNNTile(self.B, self.N, self.D)
        self.out_gate_attn = self.nntile_post_attn_scale_skip.forward(
            self.out_post_attn, self.gate_msa, self.x)
        
        self.norm_x_y = self.norm_x(torch.tensor(self.out_gate_attn, dtype=torch.float32))
        self.nntile_pre_ffn_scale_shift = ScaleShiftNNTile(self.B, self.N, self.D)
        self.x_out_pre_ffn = self.nntile_pre_ffn_scale_shift.forward(
            self.norm_x_y.detach().numpy(), self.scale_mlp, self.shift_mlp)
        
        self.nt_ff = FeedForwardNNTile(self.B, self.N, self.F, self.D, 
                                     self.W1_ff.get_array(), self.b1_ff.get_array(), 
                                     self.W2_ff.get_array(), self.b2_ff.get_array())
        self.x_out_post_ffn = self.nt_ff.forward(self.x_out_pre_ffn)
        self.nntile_post_ffn_scale_skip = ScalePlusSkipConnectionNNTile(self.B, self.N, self.D)
        self.out_gate_ffn = self.nntile_post_ffn_scale_skip.forward(
            self.x_out_post_ffn, self.gate_mlp, self.out_gate_attn)
        return self.out_gate_ffn
        

    def backward(self, grad_out_gate_ffn):
        self.grad_out_gate_ffn = grad_out_gate_ffn
        self.grad_x_out_post_ffn, self.grad_gate_mlp, self.grad_out_gate_attn = \
            self.nntile_post_ffn_scale_skip.backward(self.grad_out_gate_ffn)
        
        self.grad_x_out_pre_ffn = self.nt_ff.backward(self.grad_x_out_post_ffn)
        self.grad_norm_x_y, self.grad_scale_mlp, self.grad_shift_mlp = \
            self.nntile_pre_ffn_scale_shift.backward(self.grad_x_out_pre_ffn)
        
        norm_x_y_torch = torch.tensor(self.out_gate_attn, dtype=torch.float32, requires_grad=True)
        norm_x_y_out = self.norm_x(norm_x_y_torch)
        norm_x_y_out.backward(torch.tensor(self.grad_norm_x_y, dtype=torch.float32))
        self.grad_out_gate_attn = norm_x_y_torch.grad.detach().numpy()
        
        self.grad_out_post_attn, self.grads = self.attn_np.backward(self.grad_out_gate_attn)
        self.grad_norm_x_x, self.grad_scale_msa, self.grad_shift_msa = \
            self.nntile_pre_attn_scale_shift.backward(self.grad_out_post_attn)
        
        norm_x_x_torch = torch.tensor(self.x, dtype=torch.float32, requires_grad=True)
        norm_x_x_out = self.norm_x(norm_x_x_torch)
        norm_x_x_out.backward(torch.tensor(self.grad_norm_x_x, dtype=torch.float32))
        self.grad_x = norm_x_x_torch.grad.detach().numpy()
        
        self.grad_emb_nt, self.dW0, self.dW1, self.dW2, self.dW3, self.dW4, self.dW5, \
        self.db0, self.db1, self.db2, self.db3, self.db4, self.db5 = \
            self.mlp_nntile.backward(self.emb, self.grad_shift_msa, self.grad_scale_msa,
                                    self.grad_gate_msa, self.grad_shift_mlp, self.grad_scale_mlp,
                                    self.grad_gate_mlp)
        
        return self.grad_x, self.grad_emb_nt, {
            'W0': self.dW0, 'W1': self.dW1, 'W2': self.dW2, 'W3': self.dW3, 'W4': self.dW4, 'W5': self.dW5,
            'b0': self.db0, 'b1': self.db1, 'b2': self.db2, 'b3': self.db3, 'b4': self.db4, 'b5': self.db5,
            'W1_ff': self.nt_ff.dW1, 'b1_ff': self.nt_ff.db1, 'W2_ff': self.nt_ff.dW2, 'b2_ff': self.nt_ff.db2,
            'qkv_weight': self.grads['qkv_weight'], 'qkv_bias': self.grads['qkv_bias'],
            'out_weight': self.grads['out_weight'], 'out_bias': self.grads['out_bias']
        }