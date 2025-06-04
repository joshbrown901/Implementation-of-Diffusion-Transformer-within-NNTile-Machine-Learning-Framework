import nntile
import numpy as np
nntile_config = nntile.starpu.Config(1,0,0,0)
nntile.starpu.init()
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, TensorTraits

class AdaptiveLayerNNTile:
    def __init__(self, B, D, W1, b1, W2, b2):
        self.B = B
        self.D = D
        # Shared activation → then two separate linears
        self.W1 = nntc.from_array(W1)
        self.b1 = nntc.from_array(b1)
        self.W2 = nntc.from_array(W2)
        self.b2 = nntc.from_array(b2)

        self.dW1 = nntc.zeros_like(self.W1)
        self.db1 = nntc.zeros([D])
        self.dW2 = nntc.zeros_like(self.W2)
        self.db2 = nntc.zeros_like(self.db1)

        self.shift_mlp = nntc.zeros([self.B, self.D])
        self.scale_mlp = nntc.zeros([self.B, self.D])
        self.grad_activated1 = nntc.zeros_like(self.scale_mlp)
        self.grad_activated2 = nntc.zeros_like(self.scale_mlp)
        self.grad_activated = nntc.zeros_like(self.shift_mlp)
        self.emb_silu = nntc.zeros_like(self.shift_mlp)

    def forward(self, emb):
        self.emb = emb
        self.emb_nnt = nntc.from_array(self.emb)
        nntf.silu_forward_async(self.emb_nnt, self.emb_silu)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.W1, 0.0, self.shift_mlp, 1, 0, 0) 
        if self.b1 is not None:
            nntf.add_slice_inplace_async(1.0, self.b1, 1.0, self.shift_mlp, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.W2, 0.0, self.scale_mlp, 1, 0, 0) 
        if self.b2 is not None:
            nntf.add_slice_inplace_async(1.0, self.b2, 1.0, self.scale_mlp, 0)
        return nntc.to_numpy(self.shift_mlp), nntc.to_numpy(self.scale_mlp)

    def backward(self, grad_shift_mlp, grad_scale_mlp):
        self.grad_shift_mlp = grad_shift_mlp
        self.grad_shift_mlp_nnt = nntc.from_array(self.grad_shift_mlp)
        self.grad_scale_mlp = grad_scale_mlp
        self.grad_scale_mlp_nnt = nntc.from_array(self.grad_scale_mlp)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.grad_shift_mlp_nnt, nntile.nntile_core.notrans, self.W1, 0.0, self.grad_activated1, 1, 0, 0)         
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.grad_scale_mlp_nnt, nntile.nntile_core.notrans, self.W2, 0.0, self.grad_activated2, 1, 0, 0)
        nntf.add_async(1.0, self.grad_activated1, 1.0, self.grad_activated2, self.grad_activated)
        self.grad_emb = nntc.zeros_like(self.emb_nnt)
        nntf.silu_backward_async(self.emb_nnt, self.grad_activated, self.grad_emb)
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_shift_mlp_nnt, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.dW1, 1, 0, 0)
        nntf.sum_slice_async(1.0, self.grad_shift_mlp_nnt, 0.0, self.db1, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_scale_mlp_nnt, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.dW2, 1, 0, 0)
        nntf.sum_slice_async(1.0, self.grad_scale_mlp_nnt, 0.0, self.db2, 0)
        return nntc.to_numpy(self.grad_emb), nntc.to_numpy(self.dW1), nntc.to_numpy(self.db1), nntc.to_numpy(self.dW2), nntc.to_numpy(self.db2)