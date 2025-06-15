rom mlp_nntile import MLPNNTile
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

from mlp_nntile import MLPNNTile
#from layer_norm import manual_layer_norm
from scale_shift_nntile import ScaleShiftNNTile
#from AttentionNumpy import Attention_numpy
from scale_plus_skipconnection_nntile import ScalePlusSkipConnectionNNTile
from PointwiseFeedForwardNNTile import FeedForwardNNTile
#from TransformerLayerTorch import  TransformerLayer

import nntile
import numpy as np
import torch
import torch.nn as nn

class TransformerLayerNNTile:
    def __init__(self, B, N, F, D, W0, W1, W2, W3, W4, W5, 
                 b0, b1, b2, b3, b4, b5, W1_ff, b1_ff, W2_ff, b2_ff, 
                 bias=True):
        self.B = B
        self.N = N
        self.F = F
        self.D = D
        self.bias = bias

        # MLP weights
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

        # Feed-forward weights
        self.W1_ff = W1_ff
        self.b1_ff = b1_ff
        self.W2_ff = W2_ff
        self.b2_ff = b2_ff
        
    def forward(self, x, emb):
        self.x = x #Numpy Tensor
        self.emb = emb  
        
        self.mlp_nntile = MLPNNTile(
            self.B, self.D,
            self.W0, self.W1, self.W2, self.W3, self.W4, self.W5,
            self.b0, self.b1, self.b2, self.b3, self.b4, self.b5,
            bias=self.bias
        )
        self.shift_msa, self.scale_msa, self.gate_msa, self.shift_mlp, self.scale_mlp, self.gate_mlp = self.mlp_nntile.forward(self.emb)

        self.x_value = nntc.from_array(self.x)
        self.x_grad = nntc.from_array(np.zeros_like(self.x))
        self.x_nnt = TensorMoments(self.x_value, self.x_grad, True)
        self.layer_norm_attn, _ = LayerNormNoAffine.generate_simple(self.x_nnt, axis=2, eps=1e-6, redux=False, next_tag=100)
        nntf.fill_async(1.0, self.layer_norm_attn.gamma.value)
        nntf.clear_async(self.layer_norm_attn.beta.value)
        self.layer_norm_attn.forward_async()
        
        self.nntile_pre_attn_scale_shift = ScaleShiftNNTile(self.B, self.N, self.D)
        self.x_out_pre_attn = self.nntile_pre_attn_scale_shift.forward(nntc.to_numpy(self.layer_norm_attn.y.value), self.scale_msa, self.shift_msa)
        
        self.out_post_attn = self.x_out_pre_attn
        
        self.nntile_post_attn_scale_skip = ScalePlusSkipConnectionNNTile(self.B, self.N, self.D)
        self.out_gate_attn = self.nntile_post_attn_scale_skip.forward(
            self.out_post_attn, self.gate_msa, self.x)

        self.out_gate_value = nntc.from_array(self.out_gate_attn)
        self.out_gate_grad = nntc.from_array(np.zeros_like(self.out_gate_attn))
        self.out_gate_nnt = TensorMoments(self.out_gate_value, self.out_gate_grad, True)
        self.layer_norm_ffn, _ = LayerNormNoAffine.generate_simple(self.out_gate_nnt, axis=2, eps=1e-6, redux=False, next_tag=100)
        nntf.fill_async(1.0, self.layer_norm_ffn.gamma.value)
        nntf.clear_async(self.layer_norm_ffn.beta.value)
        self.layer_norm_ffn.forward_async()
        
        self.nntile_pre_ffn_scale_shift = ScaleShiftNNTile(self.B, self.N, self.D)
        self.x_out_pre_ffn = self.nntile_pre_ffn_scale_shift.forward(nntc.to_numpy(self.layer_norm_ffn.y.value), self.scale_mlp, self.shift_mlp)
        
        self.nt_ff = FeedForwardNNTile(self.B, self.N, self.F, self.D, 
                                     self.W1_ff, self.b1_ff, 
                                     self.W2_ff, self.b2_ff)
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

        self.layer_norm_ffn.y.grad.from_array(self.grad_norm_x_y)
        self.layer_norm_ffn.backward_async()
        self.grad_out_gate_attn = nntc.to_numpy(self.layer_norm_ffn.x.grad)

        self.grad_out_post_attn, self.grad_gate_msa, self.grad_x1 = \
        self.nntile_post_ffn_scale_skip.backward(self.grad_out_gate_attn) 

        self.grad_x_out_pre_attn = self.grad_out_post_attn

        self.grad_norm_x, self.grad_scale_msa, self.grad_shift_msa = \
            self.nntile_pre_attn_scale_shift.backward(self.grad_x_out_pre_attn)  

        self.layer_norm_attn.y.grad.from_array(self.grad_norm_x)
        self.layer_norm_attn.backward_async()
        self.grad_x = nntc.to_numpy(self.layer_norm_attn.x.grad)

        self.d_emb, self.dW0, self.dW1, self.dW2, self.dW3, self.dW4, self.dW5, self.db0, self.db1, self.db2, self.db3, self.db4, self.db5 = self.mlp_nntile.backward(self.emb, self.grad_shift_msa, self.grad_scale_msa, self.grad_gate_msa, self.grad_shift_mlp, self.grad_scale_mlp, self.grad_gate_mlp)

        return self.grad_x, self.d_emb