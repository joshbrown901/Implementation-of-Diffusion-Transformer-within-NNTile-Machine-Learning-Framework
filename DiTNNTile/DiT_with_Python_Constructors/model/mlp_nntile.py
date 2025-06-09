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




class MLPNNTile:
    def __init__(self, B, D, W0, W1, W2, W3, W4, W5, b0, b1, b2, b3, b4, b5, bias=True):
        self.bias = bias
        self.B = B
        self.D = D
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

        self.dW0 = nntc.zeros_like(self.W0)
        self.dW1 = nntc.zeros_like(self.W1)
        self.dW2 = nntc.zeros_like(self.W2)
        self.dW3 = nntc.zeros_like(self.W3)
        self.dW4 = nntc.zeros_like(self.W4)
        self.dW5 = nntc.zeros_like(self.W5)

        if self.bias:
            self.db0 = nntc.zeros_like(self.b0)
            self.db1 = nntc.zeros_like(self.b1)
            self.db2 = nntc.zeros_like(self.b2)
            self.db3 = nntc.zeros_like(self.b3)
            self.db4 = nntc.zeros_like(self.b4)
            self.db5 = nntc.zeros_like(self.b5)
        else:
            self.db0 = self.db1 = self.db2 = self.db3 = self.db4 = self.db5 = None

        self.emb_silu = nntc.zeros([self.B, self.D])
        self.grad_activated0 = nntc.zeros_like(self.emb_silu)
        self.grad_activated1 = nntc.zeros_like(self.emb_silu)
        self.grad_activated2 = nntc.zeros_like(self.emb_silu) 
        self.grad_activated3 = nntc.zeros_like(self.emb_silu) 
        self.grad_activated4 = nntc.zeros_like(self.emb_silu)
        self.grad_activated5 = nntc.zeros_like(self.emb_silu) 
        self.grad_activated = nntc.zeros_like(self.emb_silu)
        self.grad_emb = nntc.zeros_like(self.emb_silu)

        self.shift_msa = nntc.zeros_like(self.emb_silu)
        self.scale_msa = nntc.zeros_like(self.emb_silu)
        self.gate_msa = nntc.zeros_like(self.emb_silu)
        self.shift_mlp = nntc.zeros_like(self.emb_silu)
        self.scale_mlp = nntc.zeros_like(self.emb_silu)
        self.gate_mlp = nntc.zeros_like(self.emb_silu)
        

    def forward(self, emb):
        self.emb = emb
        self.emb_nnt = nntc.from_array(self.emb)
        nntf.silu_forward_async(self.emb_nnt, self.emb_silu)
        self.emb_nnt.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.W0, 0.0, self.shift_msa, 1, 0, 0) 
        self.W0.wont_use()
        if self.bias:
            nntf.add_slice_inplace_async(1.0, self.b0, 1.0, self.shift_msa, 0)
        self.b0.wont_use()
        self.shift_msa.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.W1, 0.0, self.scale_msa, 1, 0, 0)
        self.W1.wont_use()
        if self.bias:
            nntf.add_slice_inplace_async(1.0, self.b1, 1.0, self.scale_msa, 0)
        self.b1.wont_use()
        self.scale_msa.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.W2, 0.0, self.gate_msa, 1, 0, 0) 
        self.W2.wont_use()
        if self.bias:
            nntf.add_slice_inplace_async(1.0, self.b2, 1.0, self.gate_msa, 0)
        self.b2.wont_use()
        self.gate_msa.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.W3, 0.0, self.shift_mlp, 1, 0, 0)
        self.W3.wont_use()
        if self.bias:
            nntf.add_slice_inplace_async(1.0, self.b3, 1.0, self.shift_mlp, 0)
        self.b3.wont_use()
        self.shift_mlp.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.W4, 0.0, self.scale_mlp, 1, 0, 0)
        self.W4.wont_use()
        if self.bias:
            nntf.add_slice_inplace_async(1.0, self.b4, 1.0, self.scale_mlp, 0)
        self.b4.wont_use()
        self.scale_mlp.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.W5, 0.0, self.gate_mlp, 1, 0, 0)
        self.W5.wont_use()
        if self.bias:
            nntf.add_slice_inplace_async(1.0, self.b5, 1.0, self.gate_mlp, 0)
        self.b5.wont_use()
        self.gate_mlp.wont_use()
        return nntc.to_numpy(self.shift_msa), nntc.to_numpy(self.scale_msa), nntc.to_numpy(self.gate_msa), nntc.to_numpy(self.shift_mlp), nntc.to_numpy(self.scale_mlp), nntc.to_numpy(self.gate_mlp)

    def backward(self, emb, grad_shift_msa, grad_scale_msa, grad_gate_msa, grad_shift_mlp, grad_scale_mlp, grad_gate_mlp):
        self.grad_shift_msa = nntc.from_array(grad_shift_msa)
        self.grad_scale_msa = nntc.from_array(grad_scale_msa)
        self.grad_gate_msa = nntc.from_array(grad_gate_msa)
        self.grad_shift_mlp = nntc.from_array(grad_shift_mlp)
        self.grad_scale_mlp = nntc.from_array(grad_scale_mlp)
        self.grad_gate_mlp = nntc.from_array(grad_gate_mlp)
        
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.grad_shift_msa, nntile.nntile_core.notrans, self.W0, 0.0, self.grad_activated0, 1, 0, 0) 
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.grad_scale_msa, nntile.nntile_core.notrans, self.W1, 0.0, self.grad_activated1, 1, 0, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.grad_gate_msa, nntile.nntile_core.notrans, self.W2, 0.0, self.grad_activated2, 1, 0, 0) 
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.grad_shift_mlp, nntile.nntile_core.notrans, self.W3, 0.0, self.grad_activated3, 1, 0, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.grad_scale_mlp, nntile.nntile_core.notrans, self.W4, 0.0, self.grad_activated4, 1, 0, 0) 
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.grad_gate_mlp, nntile.nntile_core.notrans, self.W5, 0.0, self.grad_activated5, 1, 0, 0)
        self.W0.wont_use()
        self.W1.wont_use()
        self.W2.wont_use()
        self.W3.wont_use()
        self.W4.wont_use()
        self.W5.wont_use()
        
        nntf.add_inplace_async(1.0, self.grad_activated0, 1.0, self.grad_activated)
        self.grad_activated0.wont_use()
        nntf.add_inplace_async(1.0, self.grad_activated1, 1.0, self.grad_activated)
        self.grad_activated1.wont_use()
        nntf.add_inplace_async(1.0, self.grad_activated2, 1.0, self.grad_activated)
        self.grad_activated2.wont_use()
        nntf.add_inplace_async(1.0, self.grad_activated3, 1.0, self.grad_activated)
        self.grad_activated3.wont_use()
        nntf.add_inplace_async(1.0, self.grad_activated4, 1.0, self.grad_activated)
        self.grad_activated4.wont_use()
        nntf.add_inplace_async(1.0, self.grad_activated5, 1.0, self.grad_activated)
        self.grad_activated5.wont_use()
        
        nntf.silu_backward_async(self.emb_nnt, self.grad_activated, self.grad_emb)
        self.emb_nnt.wont_use()
        self.grad_activated.wont_use()
        self.grad_emb.wont_use()

        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_shift_msa, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.dW0, 1, 0, 0)
        self.dW0.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_scale_msa, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.dW1, 1, 0, 0)
        self.dW1.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_gate_msa, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.dW2, 1, 0, 0) 
        self.dW2.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_shift_mlp, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.dW3, 1, 0, 0)
        self.dW3.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_scale_mlp, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.dW4, 1, 0, 0)
        self.dW4.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_gate_mlp, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.dW5, 1, 0, 0)
        self.emb_silu.wont_use()
        self.dW5.wont_use()

        if self.bias:
            nntf.sum_slice_async(1.0, self.grad_shift_msa, 0.0, self.db0, 0)
            self.grad_shift_msa.wont_use()
            self.db0.wont_use()
            nntf.sum_slice_async(1.0, self.grad_scale_msa, 0.0, self.db1, 0)
            self.grad_scale_msa.wont_use()
            self.db1.wont_use()
            nntf.sum_slice_async(1.0, self.grad_gate_msa, 0.0, self.db2, 0)
            self.grad_gate_msa.wont_use()
            self.db2.wont_use()
            nntf.sum_slice_async(1.0, self.grad_shift_mlp, 0.0, self.db3, 0)
            self.grad_shift_mlp.wont_use()
            self.db3.wont_use()
            nntf.sum_slice_async(1.0, self.grad_scale_mlp, 0.0, self.db4, 0)
            self.grad_scale_mlp.wont_use()
            self.db4.wont_use()
            nntf.sum_slice_async(1.0, self.grad_gate_mlp, 0.0, self.db5, 0)
            self.grad_gate_mlp.wont_use()
            self.db5.wont_use()
        else:
            self.db0 = self.db1 = self.db2 = self.db3 = self.db4 = self.db5 = None

        return nntc.to_numpy(self.grad_emb), nntc.to_numpy(self.dW0), nntc.to_numpy(self.dW1), nntc.to_numpy(self.dW2), nntc.to_numpy(self.dW3), nntc.to_numpy(self.dW4), nntc.to_numpy(self.dW5), nntc.to_numpy(self.db0), nntc.to_numpy(self.db1), nntc.to_numpy(self.db2), nntc.to_numpy(self.db3), nntc.to_numpy(self.db4), nntc.to_numpy(self.db5)
