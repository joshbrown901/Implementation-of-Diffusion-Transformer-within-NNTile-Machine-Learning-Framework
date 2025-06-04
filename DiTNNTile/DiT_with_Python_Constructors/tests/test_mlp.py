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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest



from mlp_nntile import MLPNNTile
from mlp_numpy import MLPNumpyUnrolled
from mlp_torch import MLPTorch


def relative_error(a, b, eps=1e-8):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + eps)


@pytest.mark.parametrize("B, D", [(3, 4), (5, 7)])
def test_mlp_nntile_vs_numpy_and_torch(B, D):
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize weights and biases
    W_np = [np.random.randn(D, D).astype(np.float32) for _ in range(6)]
    b_np = [np.random.randn(D).astype(np.float32) for _ in range(6)]

    # NumPy MLP
    mlp_np = MLPNumpyUnrolled(D, bias=True)
    mlp_np.W0, mlp_np.W1, mlp_np.W2, mlp_np.W3, mlp_np.W4, mlp_np.W5 = W_np
    mlp_np.b0, mlp_np.b1, mlp_np.b2, mlp_np.b3, mlp_np.b4, mlp_np.b5 = b_np

    # PyTorch MLP
    mlp_torch = MLPTorch(D, bias=True)
    with torch.no_grad():
        mlp_torch.W.copy_(torch.from_numpy(np.vstack(W_np)))
        if mlp_torch.b is not None:
            mlp_torch.b.copy_(torch.from_numpy(np.hstack(b_np)))

    # NNTile MLP - make sure to pass weights in correct order
    mlp_nntile = MLPNNTile(B, D, *W_np, *b_np, bias=True)

    # Input embedding
    emb_np = np.random.randn(B, D).astype(np.float32)
    emb_torch = torch.tensor(emb_np, requires_grad=True)

    # Forward pass
    # NumPy
    out0_np, out1_np, out2_np, out3_np, out4_np, out5_np = mlp_np.forward(emb_np)
    
    # PyTorch
    out_torch = mlp_torch(emb_torch)
    out_torch_np = [o.detach().numpy() for o in out_torch]
    
    # NNTile 
    out_nt = mlp_nntile.forward(emb_np)
    
    # Compare outputs one by one
    for i in range(6):
        err_np = relative_error(out_nt[i], [out0_np, out1_np, out2_np, out3_np, out4_np, out5_np][i])
        err_torch = relative_error(out_nt[i], out_torch_np[i])
        print(f"Output {i} NNTile vs NumPy error:", err_np)
        print(f"Output {i} NNTile vs Torch error:", err_torch)
        assert err_np < 1e-5
        assert err_torch < 1e-5

    # Backward pass
    grads_np = [np.random.randn(B, D).astype(np.float32) for _ in range(6)]
    
    # NumPy backward
    grad_emb_np, grad_W_np, grad_b_np = mlp_np.backward(emb_np, *grads_np)
    
    # PyTorch backward
    mlp_torch.zero_grad()
    loss = sum((out_torch[i] * torch.tensor(grads_np[i])).sum() for i in range(6))
    loss.backward()
    grad_emb_torch = emb_torch.grad.detach().numpy()
    grad_W_torch = mlp_torch.W.grad.detach().numpy()
    grad_b_torch = mlp_torch.b.grad.detach().numpy() if mlp_torch.b is not None else None
    
    # NNTile backward
    grad_emb_nt, dW0, dW1, dW2, dW3, dW4, dW5, db0, db1, db2, db3, db4, db5 = mlp_nntile.backward(emb_np, *grads_np)
    grad_W_nt = np.vstack([dW0, dW1, dW2, dW3, dW4, dW5])
    grad_b_nt = np.hstack([db0, db1, db2, db3, db4, db5]) if db0 is not None else None
    
    # Compare gradients
    print("Grad emb rel error (NNTile vs NumPy):", relative_error(grad_emb_nt, grad_emb_np))
    print("Grad W rel error (NNTile vs NumPy):", relative_error(grad_W_nt, np.vstack(grad_W_np)))
    if grad_b_nt is not None:
        print("Grad b rel error (NNTile vs NumPy):", relative_error(grad_b_nt, np.hstack(grad_b_np)))
    
    assert relative_error(grad_emb_nt, grad_emb_np) < 1e-5
    assert relative_error(grad_W_nt, np.vstack(grad_W_np)) < 1e-5
    if grad_b_nt is not None:
        assert relative_error(grad_b_nt, np.hstack(grad_b_np)) < 1e-5

test_mlp_nntile_vs_numpy_and_torch(3, 4)

test_mlp_nntile_vs_numpy_and_torch(5, 7)
