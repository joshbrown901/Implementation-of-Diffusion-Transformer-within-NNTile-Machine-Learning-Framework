import torch
import nntile
nntile_config = nntile.starpu.Config(1,0,0,0)
nntile.starpu.init()
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, TensorTraits
from typing import List, Union
import torch.nn as nn
import numpy as np
from nntile.tensor import TensorMoments, to_numpy, from_array, fill_async, clear_async, sum_slice_async, norm_slice_async, hypot_scalar_inverse_async, prod_slice_async, add_slice_async, add_inplace_async, add_slice_inplace_async, sumprod_slice_async
from nntile.layer.layer_norm import LayerNorm

# Custom LayerNorm class to skip affine transformation
class LayerNormNoAffine(LayerNorm):
    def __init__(self, x, y, gamma, beta, tmp_y_value, tmp_y_grad, mean, inv_stddev, axis, eps, redux=False):
        super().__init__(x, y, gamma, beta, tmp_y_value, tmp_y_grad, mean, inv_stddev, axis, eps, redux)
        # Disable gradient computations for gamma and beta
        self.gamma.grad_required = False
        self.beta.grad_required = False

    def forward_async(self):
        # Get means over given axis
        sum_slice_async(
            1.0 / self.l,
            self.x.value,
            0.0,
            self.mean,
            self.axis,
            redux=self.redux,
        )
        # tmp_y_value = x - mean
        add_slice_async(
            -1.0, self.mean, 1.0, self.x.value, self.tmp_y_value, self.axis
        )
        # mean can be offloaded from GPU
        self.mean.wont_use()
        # x can be offloaded from GPU
        self.x.value.wont_use()
        # Compute standard deviation
        norm_slice_async(
            1.0 / self.l**0.5,
            self.tmp_y_value,
            0.0,
            self.inv_stddev,
            self.axis,
            redux=self.redux,
        )
        hypot_scalar_inverse_async(self.eps, 1.0, self.inv_stddev)
        # Normalize input: y = (x - mean) / stddev
        prod_slice_async(self.inv_stddev, 1.0, self.tmp_y_value, self.axis)
        # Copy tmp_y_value to y.value (no gamma or beta)
        add_inplace_async(1.0, self.tmp_y_value, 0.0, self.y.value)
        # Clean up
        self.inv_stddev.wont_use()
        self.tmp_y_value.wont_use()
        self.y.value.wont_use()

    def backward_async(self):
        # Skip gamma and beta gradient computations
        # Define gradient over normalized input: tmp_y_grad = dy / sigma
        prod_slice_async(self.inv_stddev, 1.0, self.y.grad, self.tmp_y_grad, self.axis)
        # Get mean of tmp_y_grad over the given axis
        sum_slice_async(
            1.0 / self.l,
            self.tmp_y_grad,
            0.0,
            self.mean,
            self.axis,
            redux=self.redux,
        )
        # Subtract mean from tmp_y_grad: tmp_y_grad -= mean(dy / sigma)
        add_slice_inplace_async(
            -1.0, self.mean, 1.0, self.tmp_y_grad, self.axis
        )
        # Get mean of product of tmp_y_grad and tmp_y_value over the given axis
        sumprod_slice_async(
            -1.0 / self.l,
            self.tmp_y_grad,
            self.tmp_y_value,
            0.0,
            self.mean,
            self.axis,
            redux=self.redux,
        )
        # Multiply tmp_y_value by the mean
        prod_slice_async(self.mean, 1.0, self.tmp_y_value, self.axis)
        # Add tmp_y_grad to tmp_y_value
        add_inplace_async(1.0, self.tmp_y_grad, 1.0, self.tmp_y_value)
        # tmp_y_grad can be deleted
        self.tmp_y_grad.invalidate_submit()
        # tmp_y_value now contains the input gradient
        add_inplace_async(1.0, self.tmp_y_value, 1.0, self.x.grad)
        # Clean up
        self.mean.invalidate_submit()
        self.inv_stddev.invalidate_submit()
        self.tmp_y_value.invalidate_submit()
        self.x.grad.wont_use()
