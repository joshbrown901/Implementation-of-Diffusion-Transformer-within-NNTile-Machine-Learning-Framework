import nntile
import numpy as np
nntile_config = nntile.starpu.Config(1,0,0,0)
nntile.starpu.init()
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, TensorTraits


class FeedForwardNNTile:
    def __init__(self, B, N, F, D, W1, b1, W2, b2):
        self.B = B
        self.N = N
        self.F = F
        self.D = D

        self.W1 = nntc.from_array(W1)
        self.b1 = nntc.from_array(b1)
        self.W2 = nntc.from_array(W2)
        self.b2 = nntc.from_array(b2)

        self.dW1 = nntc.zeros_like(self.W1)
        self.db1 = nntc.zeros_like(self.b1)
        self.dW2 = nntc.zeros_like(self.W2)
        self.db2 = nntc.zeros_like(self.b2)

        self.x11 = nntc.zeros([self.B, self.N, self.F])
        self.x1 = nntc.zeros_like(self.x11)
        self.a1 = nntc.zeros_like(self.x11)
        self.out1 = nntc.zeros([self.B, self.N, self.D])
        self.out = nntc.zeros_like(self.out1)

    def forward(self, x):
        self.x = x
        self.x_nnt = nntc.from_array(self.x)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.x_nnt, nntile.nntile_core.trans, self.W1, 0.0, self.x11, 1, 0, 0)
        nntf.add_fiber_async(1.0, self.b1, 1.0, self.x11, self.x1, 2, 0)
        nntf.gelutanh_async(self.x1, self.a1)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.a1, nntile.nntile_core.trans, self.W2, 0.0, self.out1, 1, 0, 0)
        nntf.add_fiber_async(1.0, self.b2, 1.0, self.out1, self.out, 2, 0)
        return nntc.to_numpy(self.out)

    def backward(self, grad_out):  
        self.grad_out = nntc.from_array(grad_out)
        self.grad_out1 = nntc.from_array(nntc.to_numpy(self.grad_out).reshape(-1, self.D))
        self.grad_a1 = nntc.from_array(nntc.to_numpy(self.a1).reshape(-1, self.F))
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_out1, nntile.nntile_core.notrans, self.grad_a1, 0.0, self.dW2, 1, 0, 0)
        nntf.sum_fiber_async(1.0, self.grad_out, 0.0, self.db2, 2, 0)
        self.da1 = nntc.zeros_like(self.x11)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.grad_out, nntile.nntile_core.notrans, self.W2, 0.0, self.da1, 1, 0, 0)
        self.dx1 = nntc.zeros_like(self.x11)
        nntf.gelutanh_backward_async(self.x1, self.da1, self.dx1)
        self.grad_x1 = nntc.from_array(nntc.to_numpy(self.dx1).reshape(-1, self.F))
        self.x_flat = nntc.from_array(self.x.reshape(-1, self.D))
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_x1, nntile.nntile_core.notrans, self.x_flat, 0.0, self.dW1, 1, 0, 0)        
        nntf.sum_fiber_async(1.0, self.dx1, 0.0, self.db1, 2, 0) 
        self.dx = nntc.zeros_like(self.x_nnt)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.dx1, nntile.nntile_core.notrans, self.W1, 0.0, self.dx, 1, 0, 0)
        return nntc.to_numpy(self.dx)