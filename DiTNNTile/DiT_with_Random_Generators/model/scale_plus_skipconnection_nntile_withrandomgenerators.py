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



class ScalePlusSkipConnectionNumpy:
    @staticmethod
    def forward(x: np.ndarray, scale: np.ndarray, shift: np.ndarray):
        # scale, shift are (B, D); broadcast to (B, N, D)
        scale_b = (1.0 + scale)[:, None, :]    # shape (B,1,D) → broadcast
        y = x * scale_b + shift
        return y

    @staticmethod
    def backward(x: np.ndarray, scale: np.ndarray, shift: np.ndarray, grad_y: np.ndarray):
        s = (1.0 + scale)[:, None, :]           # shape (B,1,D)
        grad_x = grad_y * s
        # accumulate over N
        grad_scale = np.sum(grad_y * x, axis=1) # shape (B,D)
        #grad_shift = grad_y     # shape (B,D)
        return grad_x, grad_scale #grad_shift

class ScalePlusSkipConnectionTorch(nn.Module):
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # scale, shift are (B,D); avoid any in-place ops
        scale_mod = scale + 1.0                   # out-of-place
        y = x * scale_mod.unsqueeze(1) + shift
        return y


class scale_plus_skipconnection(BaseLayer):
    def __init__(self, x: TensorMoments, scale: TensorMoments, shift: TensorMoments, y: TensorMoments, tmp: Tensor, scale_new: Tensor, grad_scale_applied: Tensor, tmp1: Tensor):
        self.x = x
        self.y = y
        self.scale = scale
        self.shift = shift
        self.tmp = tmp
        self.scale_new = scale_new
        self.grad_scale_applied = grad_scale_applied
        self.tmp1 = tmp1
        
        super().__init__([x, scale, shift], [y], [], [tmp, scale_new, tmp1])

    @staticmethod
    def generate_simple(x: TensorMoments, scale: TensorMoments, shift: TensorMoments, x_traits: TensorTraits, x_distr, next_tag:int):
        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        x_distr = x.value.distribution
        scale_traits = TensorTraits(scale.value.shape, scale.value.basetile_shape)
        scale_distr = scale.value.distribution
        shift_traits = TensorTraits(shift.value.shape, shift.value.basetile_shape)
        shift_distr = shift.value.distribution
        y_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        y_distr = [0] * y_traits.grid.nelems
        y_value = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_value.next_tag
        y_grad = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_grad.next_tag
        y = TensorMoments(y_value, y_grad, True)
        tmp = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = tmp.next_tag
        scale_new = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = scale_new.next_tag
        grad_scale_applied = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = grad_scale_applied.next_tag
        tmp1 = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = tmp1.next_tag
        return scale_plus_skipconnection(x, scale, shift, y, tmp, scale_new, grad_scale_applied, tmp1), next_tag

    def forward_async(self):
        nntf.fill_async(1.0, self.tmp)
        nntf.fill_async(0.0, self.scale_new)
        nntf.add_slice_async(1.0, self.scale.value, 1.0, self.tmp, self.scale_new, 1)
        self.scale.value.wont_use()
        self.tmp.wont_use()
        nntf.prod_inplace_async(self.x.value, self.scale_new)
        self.x.value.wont_use()
        nntf.add_async(1.0, self.shift.value, 1.0, self.scale_new, self.y.value)
        self.shift.value.wont_use()
        self.scale_new.wont_use()
        self.y.value.wont_use()

    def backward_async(self):
        nntf.fill_async(0.0, self.tmp1)
        nntf.add_slice_async(1.0, self.scale.value, 1.0, self.tmp, self.tmp1, 1)
        self.scale.value.wont_use()
        self.tmp.wont_use()       
        nntf.prod_async(self.y.grad, self.tmp1, self.x.grad)
        self.tmp1.wont_use()
        self.x.grad.wont_use()
        nntf.prod_async(self.y.grad, self.x.value, self.grad_scale_applied)
        self.y.grad.wont_use()
        self.x.value.wont_use()
        nntf.sum_slice_async(1.0, self.grad_scale_applied, 0.0, self.scale.grad, 1)
        self.grad_scale_applied.wont_use()
        self.scale.grad.wont_use()

    @classmethod
    def from_torch(cls, torch_module: ScalePlusSkipConnectionTorch,
                   x: TensorMoments,
                   next_tag: int):
        
        scale_np = torch_module.scale.data.cpu().numpy()
        shift_np = torch_module.shift.data.cpu().numpy()

        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        x_distr = x.value.distribution

        # build scale TensorMoments
        scale_value = nntc.from_array(scale_np)
        scale_grad = nntc.zeros_like(scale_value)
        scale_tm = TensorMoments(scale_value, scale_grad, True)
        
        # build shift TensorMoments
        shift_value = nntc.from_array(shift_np)
        shift_grad = nntc.zeros_like(shift_value)
        shift_tm = TensorMoments(shift_value, shift_grad, True)

        layer, next_tag = cls.generate_simple(x, scale_tm, shift_tm, x_traits, x_distr, next_tag)

        return layer, next_tag

    def to_torch(self) -> ScalePlusSkipConnectionTorch:
        torch_mod = ScalePlusSkipConnectionTorch()
        scale_arr = nntc.to_numpy(self.scale.value)
        shift_arr = nntc.to_numpy(self.shift.value)
        torch_mod.scale = nn.Parameter(torch.tensor(scale_arr, dtype=torch.float32))
        torch_mod.shift = nn.Parameter(torch.tensor(shift_arr, dtype=torch.float32))
        return torch_mod

    def to_torch_with_grads(self) -> ScalePlusSkipConnectionTorch:
        torch_mod = self.to_torch()
        torch_mod.scale.grad = torch.tensor(nntc.to_numpy(self.scale.grad), dtype=torch.float32)
        torch_mod.shift.grad = torch.tensor(nntc.to_numpy(self.shift.grad), dtype=torch.float32)
        return torch_mod