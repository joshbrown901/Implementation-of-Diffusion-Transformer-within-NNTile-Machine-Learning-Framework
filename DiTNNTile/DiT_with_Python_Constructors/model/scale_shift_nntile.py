import nntile
import numpy as np
nntile_config = nntile.starpu.Config(1,0,0,0)
nntile.starpu.init()
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, TensorTraits


class ScaleShiftNNTile:
    def __init__(self, B, N, D):
        self.B = B
        self.N = N
        self.D = D
        self.tmp = nntc.ones([self.B, self.N, self.D])
        self.scale_new = nntc.zeros_like(self.tmp)
        self.grad_scale_applied = nntc.zeros_like(self.tmp)
        self.tmp1 = nntc.zeros_like(self.tmp)
        self.y_nnt = nntc.zeros_like(self.tmp)

    def forward(self, x, scale, shift):
        self.B, self.N, self.D = x.shape
        self.x = x
        self.x_nnt = nntc.from_array(self.x)
        self.scale = scale
        self.scale_nnt = nntc.from_array(self.scale)
        self.shift = shift
        self.shift_nnt = nntc.from_array(self.shift)
        nntf.add_slice_async(1.0, self.scale_nnt, 1.0, self.tmp, self.scale_new, 1)
        self.scale_nnt.wont_use()
        self.tmp.wont_use()
        nntf.prod_inplace_async(self.x_nnt, self.scale_new)
        self.x_nnt.wont_use()
        nntf.add_slice_async(1.0, self.shift_nnt, 1.0, self.scale_new, self.y_nnt, 1)
        self.shift_nnt.wont_use()
        self.scale_new.wont_use()
        self.y_nnt.wont_use()
        return nntc.to_numpy(self.y_nnt)

    def backward(self, grad_y):
        self.grad_y = grad_y
        self.grad_y_nnt = nntc.from_array(self.grad_y)
        nntf.add_slice_async(1.0, self.scale_nnt, 1.0, self.tmp, self.tmp1, 1)
        self.scale_nnt.wont_use()
        self.tmp.wont_use()
        self.grad_x = nntc.zeros_like(self.x_nnt)
        nntf.prod_async(self.grad_y_nnt, self.tmp1, self.grad_x)
        self.grad_x.wont_use()
        self.tmp1.wont_use()
        nntf.prod_async(self.grad_y_nnt, self.x_nnt, self.grad_scale_applied)
        self.x_nnt.wont_use()
        self.grad_scale = nntc.zeros_like(self.scale_nnt)
        nntf.sum_slice_async(1.0, self.grad_scale_applied, 0.0, self.grad_scale, 1)
        self.grad_scale_applied.wont_use()
        self.grad_scale.wont_use()
        self.grad_shift = nntc.zeros_like(self.shift_nnt)
        nntf.sum_slice_async(1.0, self.grad_y_nnt, 0.0, self.grad_shift, 1)
        self.grad_y_nnt.wont_use()
        self.grad_shift.wont_use()
        return nntc.to_numpy(self.grad_x), nntc.to_numpy(self.grad_scale), nntc.to_numpy(self.grad_shift)
