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
import pytest
from PointwiseFeedForwardNumpy import FeedForwardNumpy
from PointwiseFeedForwardTorch import FeedForwardTorch
from PointwiseFeedForwardNNTile import FeedForwardNNTile


def max_rel_error(a, b, eps=1e-8):
    return np.max(np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + eps))

@pytest.mark.parametrize("B,N,D,F", [
    (2, 4, 8, 16),
    (1, 3, 5, 10),
])
def test_nntile_vs_numpy_and_torch(B, N, D, F):
    np.random.seed(0)
    torch.manual_seed(0)


    x_np      = np.random.randn(B, N, D).astype(np.float32)
    grad_out  = np.random.randn(B, N, D).astype(np.float32)


    np_ff  = FeedForwardNumpy(D, F)
    out_np = np_ff.forward(x_np)
    W1_np = np_ff.W1
    dx_np  = np_ff.backward(grad_out)
    dW1_np, db1_np = np_ff.dW1, np_ff.db1
    dW2_np, db2_np = np_ff.dW2, np_ff.db2


    torch_ff = FeedForwardTorch(D, F)
    with torch.no_grad():
        torch_ff.fc1.weight.copy_(torch.from_numpy(np_ff.W1))
        torch_ff.fc1.bias.copy_(  torch.from_numpy(np_ff.b1))
        torch_ff.fc2.weight.copy_(torch.from_numpy(np_ff.W2))
        torch_ff.fc2.bias.copy_(  torch.from_numpy(np_ff.b2))

    x_torch = torch.tensor(x_np, requires_grad=True)
    out_torch = torch_ff(x_torch)
    out_torch_np = out_torch.detach().numpy()
    out_torch.backward(torch.tensor(grad_out))
    dx_torch = x_torch.grad.detach().numpy()
    dW1_torch = torch_ff.fc1.weight.grad.detach().numpy()
    db1_torch = torch_ff.fc1.bias.grad.detach().numpy()
    dW2_torch = torch_ff.fc2.weight.grad.detach().numpy()
    db2_torch = torch_ff.fc2.bias.grad.detach().numpy()


    nt_ff = FeedForwardNNTile(B, N, F, D, np_ff.W1, np_ff.b1, np_ff.W2, np_ff.b2)
    out_nt = nt_ff.forward(x_np)
    W1_nt = nntc.to_numpy(nt_ff.W1)
    dx_nt  = nt_ff.backward(grad_out)
    dW1_nt = nntc.to_numpy(nt_ff.dW1)
    db1_nt = nntc.to_numpy(nt_ff.db1)
    dW2_nt = nntc.to_numpy(nt_ff.dW2)
    db2_nt = nntc.to_numpy(nt_ff.db2)

 
    errs = {
        "x1 Numpy vs NNTile":        max_rel_error(W1_nt,       W1_np),
        "forward NumPy vs Torch":    max_rel_error(out_np,     out_torch_np),
        "forward NumPy vs NNTile":   max_rel_error(out_np,     out_nt),
        "forward NNTile vs Torch":   max_rel_error(out_nt,     out_torch_np),
        "backward dx NumPy vs Torch": max_rel_error(dx_np,      dx_torch),
        "backward dx NumPy vs NNTile":max_rel_error(dx_np,      dx_nt),
        "backward dx NNTile vs Torch":max_rel_error(dx_nt,      dx_torch),
        "dW1 NumPy vs Torch":        max_rel_error(dW1_np,     dW1_torch),
        "dW1 NumPy vs NNTile":       max_rel_error(dW1_np,     dW1_nt),
        "dW1 NNTile vs Torch":       max_rel_error(dW1_nt,     dW1_torch),
        "db1 NumPy vs Torch":        max_rel_error(db1_np,     db1_torch),
        "db1 NumPy vs NNTile":       max_rel_error(db1_np,     db1_nt),
        "db1 NNTile vs Torch":       max_rel_error(db1_nt,     db1_torch),
        "dW2 NumPy vs Torch":        max_rel_error(dW2_np,     dW2_torch),
        "dW2 NumPy vs NNTile":       max_rel_error(dW2_np,     dW2_nt),
        "dW2 NNTile vs Torch":       max_rel_error(dW2_nt,     dW2_torch),
        "db2 NumPy vs Torch":        max_rel_error(db2_np,     db2_torch),
        "db2 NumPy vs NNTile":       max_rel_error(db2_np,     db2_nt),
        "db2 NNTile vs Torch":       max_rel_error(db2_nt,     db2_torch),
    }

    print("\nRelative errors:")
    for name, err in errs.items():
        print(f" {name}: {err:.2e}")


test_nntile_vs_numpy_and_torch(2, 3, 5, 8)