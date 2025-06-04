from patch_embedding_torch import PatchEmbedding
import torch
import torch.nn as nn
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test
if __name__ == '__main__':
    B, C, H, W = 2, 3, 32, 32
    ph, pw = 4, 4
    hidden = 64
    torch.manual_seed(0)
    
    x = torch.randn(B, C, H, W, requires_grad=True)
    model = PatchEmbedding(H, W, C, ph, pw, hidden)

    # Forward test
    out = model(x)
    print("Output shape (forward):", out.shape)

    # # Backward test
    # grad_output = torch.randn_like(out)
    # grad_x = model.backward(grad_output)
    # print("Gradient w.r.t input shape (backward):", grad_x.shape)

    # Check autograd consistency
    out_autograd = model(x).sum()
    out_autograd.backward()
    print("Autograd grad shape:", x.grad.shape)
    # print("Autograd grad vs manual grad diff:", torch.norm(x.grad - grad_x))