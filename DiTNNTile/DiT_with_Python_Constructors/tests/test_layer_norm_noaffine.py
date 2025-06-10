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
from layer_norm_noaffine_nntile import LayerNormNoAffine


def test_layernorm():
    # Dummy data for 3D tensor
    x_data = torch.randn(32, 128, 64).numpy()  # 3D tensor with shape (32, 128, 64)
    x_value = from_array(x_data)
    x_grad = from_array(np.zeros_like(x_data))
    x = TensorMoments(x_value, x_grad, True)
    
    # Create LayerNorm layer, normalizing over the last axis (axis=2)
    layer, _ = LayerNormNoAffine.generate_simple(x, axis=2, eps=1e-5, redux=False, next_tag=100)
    
    # Initialize gamma and beta (not used, but required by LayerNorm)
    fill_async(1.0, layer.gamma.value)  # gamma = ones (neutral)
    clear_async(layer.beta.value)  # beta = zeros (neutral)
    
    # Run forward pass
    layer.forward_async()
    # sync_all()  # Uncomment if synchronization is needed
    
    # Get NNTile result
    nntile_result = to_numpy(layer.y.value)
    
    # Run equivalent PyTorch LayerNorm with elementwise_affine=False
    torch_layernorm = torch.nn.LayerNorm(64, eps=1e-5, elementwise_affine=False)
    x_torch = torch.tensor(x_data, requires_grad=True)
    torch_result = torch_layernorm(x_torch)
    
    # Save forward result
    torch_result_np = torch_result.detach().numpy()
    
    # Calculate relative error for forward
    forward_numerator = np.linalg.norm(nntile_result - torch_result_np)
    forward_denominator = np.linalg.norm(torch_result_np)
    forward_rel_error = forward_numerator / forward_denominator
    print(f"Forward relative error: {forward_rel_error:.6e}")
    
    # Backward pass: Set up a dummy loss gradient
    dy_data = torch.randn(32, 128, 64).numpy()  # Gradient of loss w.r.t. output
    layer.y.grad.from_array(dy_data)
    layer.backward_async()
    # sync_all()  # Uncomment if synchronization is needed
    
    # Get NNTile input gradient
    nntile_grad_x = to_numpy(layer.x.grad)
    
    # PyTorch backward pass
    dy_torch = torch.tensor(dy_data)
    torch_result.backward(dy_torch)
    torch_grad_x = x_torch.grad.detach().numpy()
    
    # Calculate relative error for input gradient
    backward_numerator = np.linalg.norm(nntile_grad_x - torch_grad_x)
    backward_denominator = np.linalg.norm(torch_grad_x)
    backward_rel_error = backward_numerator / backward_denominator
    print(f"Backward relative error: {backward_rel_error:.6e}")

if __name__ == "__main__":
    test_layernorm()
