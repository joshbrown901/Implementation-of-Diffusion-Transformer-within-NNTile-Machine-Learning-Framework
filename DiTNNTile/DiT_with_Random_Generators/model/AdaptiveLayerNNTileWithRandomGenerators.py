import nntile
import numpy as np
nntile_config = nntile.starpu.Config(1,0,0,0)
nntile.starpu.init()
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, TensorTraits

# Reference NumPy implementation
class AdaptiveLayerNumpy:
    def __init__(self, B, D, W1, b1, W2, b2):
        self.B, self.D = B, D
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2

    @staticmethod
    def silu(x):
        sig = 1 / (1 + np.exp(-x))
        return x * sig

    @staticmethod
    def dsilu(x):
        sig = 1 / (1 + np.exp(-x))
        return sig + x * sig * (1 - sig)

    def forward(self, x):
        self.x = x
        self.a = self.silu(x)
        self.y1 = self.a @ self.W1.T + self.b1
        self.y2 = self.a @ self.W2.T + self.b2
        return self.y1, self.y2

    def backward(self, grad_y1, grad_y2):
        grad_a1 = grad_y1 @ self.W1
        grad_a2 = grad_y2 @ self.W2
        grad_a = grad_a1 + grad_a2
        grad_x = grad_a * self.dsilu(self.x)
        grad_W1 = grad_y1.T @ self.a
        grad_b1 = grad_y1.sum(axis=0)
        grad_W2 = grad_y2.T @ self.a
        grad_b2 = grad_y2.sum(axis=0)
        return grad_x, grad_W1, grad_b1, grad_W2, grad_b2

# Reference PyTorch implementation
class AdaptiveLayerTorch(torch.nn.Module):
    def __init__(self, D):
        super().__init__()
        self.act = torch.nn.SiLU()
        self.fc1 = torch.nn.Linear(D, D, bias=True)
        self.fc2 = torch.nn.Linear(D, D, bias=True)

    def forward(self, x):
        a = self.act(x)
        return self.fc1(a), self.fc2(a)

class AdaptiveLayer(BaseLayer):
     emb: TensorMoments
     w1: TensorMoments 
     b1: TensorMoments 
     w2: TensorMoments 
     b2: TensorMoments    
     shift_mlp: TensorMoments 
     scale_mlp: TensorMoments    
     grad_activated1: Tensor
     grad_activated2: Tensor
     grad_activated: Tensor
     emb_silu: Tensor 
    
     def __init__(self,
         emb: TensorMoments, 
         w1: TensorMoments, 
         b1: TensorMoments, 
         w2: TensorMoments, 
         b2: TensorMoments,                
         shift_mlp: TensorMoments, 
         scale_mlp: TensorMoments,                   
         grad_activated1: Tensor,
         grad_activated2: Tensor,
         grad_activated: Tensor,
         emb_silu: Tensor
        ):

        super().__init__(
            [emb],
            [shift_mlp, scale_mlp],
            [],
            [grad_activated1, grad_activated2, grad_activated, emb_silu],
        )
        
        self.emb = emb
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.shift_mlp = shift_mlp
        self.scale_mlp = scale_mlp
        self.grad_activated1 = grad_activated1
        self.grad_activated2 = grad_activated2
        self.grad_activated = grad_activated                
        self.emb_silu = emb_silu

     @staticmethod
     def generate_simple(
        emb: TensorMoments, 
        w1: TensorMoments,
        b1: TensorMoments,                       
        w2: TensorMoments, 
        b2: TensorMoments, 
        emb_traits: TensorTraits, 
        emb_distr, 
        next_tag:int
    ):

        [B, D] = emb.value.shape
        [B_tile, D_tile] = emb.value.basetile_shape

        emb_traits = TensorTraits([B, D], [B_tile, D_tile])
        emb_distr = emb.value.distribution

        w1_traits = TensorTraits([D, D], [D_tile, D_tile])
        w1_distr = w1.value.distribution

        b1_traits = TensorTraits([D], [D_tile])
        b1_distr = b1.value.distribution       

        w2_traits = TensorTraits([D, D], [D_tile, D_tile])
        w2_distr = w2.value.distribution 

        b2_traits = TensorTraits([D], [D_tile])
        b2_distr = b2.value.distribution 

        shift_mlp_traits = TensorTraits([B, D], [B_tile, D_tile])
        shift_mlp_distr = [0] * shift_mlp_traits.grid.nelems
        shift_mlp_value = type(emb.value)(shift_mlp_traits, shift_mlp_distr, next_tag)
        next_tag = shift_mlp_value.next_tag
        shift_mlp_grad = type(emb.value)(shift_mlp_traits, shift_mlp_distr, next_tag)
        next_tag = shift_mlp_grad.next_tag
        shift_mlp = TensorMoments(shift_mlp_value, shift_mlp_grad, True)    

        scale_mlp_traits = TensorTraits([B, D], [B_tile, D_tile])
        scale_mlp_distr = [0] * scale_mlp_traits.grid.nelems
        scale_mlp_value = type(emb.value)(scale_mlp_traits, scale_mlp_distr, next_tag)
        next_tag = scale_mlp_value.next_tag
        scale_mlp_grad = type(emb.value)(scale_mlp_traits, scale_mlp_distr, next_tag)
        next_tag = scale_mlp_grad.next_tag
        scale_mlp = TensorMoments(scale_mlp_value, scale_mlp_grad, True)

        grad_activated1 = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = grad_activated1.next_tag

        grad_activated2 = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = grad_activated2.next_tag

        grad_activated = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = grad_activated.next_tag        

        emb_silu = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = emb_silu.next_tag 

        layer = AdaptiveLayer(
             emb, 
             w1, 
             b1, 
             w2, 
             b2,                
             shift_mlp, 
             scale_mlp,                   
             grad_activated1,
             grad_activated2,
             grad_activated,
             emb_silu
        )

        return (layer, next_tag)

     def forward_async(self):
        nntf.silu_forward_async(self.emb.value, self.emb_silu)       
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.w1.value, 0.0, self.shift_mlp.value, 1, 0, 0) 
        nntf.add_slice_inplace_async(1.0, self.b1.value, 1.0, self.shift_mlp.value, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.w2.value, 0.0, self.scale_mlp.value, 1, 0, 0) 
        nntf.add_slice_inplace_async(1.0, self.b2.value, 1.0, self.scale_mlp.value, 0)

     def backward_async(self):
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.shift_mlp.grad, nntile.nntile_core.notrans, self.w1.value, 0.0, self.grad_activated1, 1, 0, 0)         
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.scale_mlp.grad, nntile.nntile_core.notrans, self.w2.value, 0.0, self.grad_activated2, 1, 0, 0)
        nntf.add_async(1.0, self.grad_activated1, 1.0, self.grad_activated2, self.grad_activated)      
        nntf.silu_backward_async(self.emb.value, self.grad_activated, self.emb.grad)
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.shift_mlp.grad, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.w1.grad, 1, 0, 0)
        nntf.sum_slice_async(1.0, self.shift_mlp.grad, 0.0, self.b1.grad, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.scale_mlp.grad, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.w2.grad, 1, 0, 0)
        nntf.sum_slice_async(1.0, self.scale_mlp.grad, 0.0, self.b2.grad, 0)

     @classmethod
     def from_torch(cls,
        torch_layer: AdaptiveLayerTorch, 
        emb: TensorMoments,
        w1: TensorMoments,
        b1: TensorMoments,
        w2: TensorMoments,
        b2: TensorMoments,
        next_tag: int
    ):
        # Create the NNTile layer
        B, D = emb.value.shape
        emb_traits = TensorTraits([B, D], emb.value.basetile_shape)
        emb_distr = emb.value.distribution
        
        nntile_layer, next_tag = cls.generate_simple(
            emb, w1, b1, w2, b2, emb_traits, emb_distr, next_tag
        )
        
        # Copy weights and biases from PyTorch layer
        nntile_layer.w1.value.from_array(
            torch_layer.fc1.weight.data.cpu().detach().numpy().T)  # Transpose for correct layout
        nntile_layer.b1.value.from_array(
            torch_layer.fc1.bias.data.cpu().detach().numpy().T)
        nntile_layer.w2.value.from_array(
            torch_layer.fc2.weight.data.cpu().detach().numpy().T)  # Transpose for correct layout
        nntile_layer.b2.value.from_array(
            torch_layer.fc2.bias.data.cpu().detach().numpy().T)
        
        return nntile_layer, next_tag
    
     def to_torch(self) -> AdaptiveLayerTorch:
        torch_layer = AdaptiveLayerTorch(self.emb.value.shape[1])
        
        # Set weights and biases (transpose weights back to PyTorch format)
        torch_layer.fc1.weight.data = torch.tensor(
            nntc.to_numpy(self.w1.value).T,  # Transpose back
            requires_grad=True
        )
        torch_layer.fc1.bias.data = torch.tensor(
            nntc.to_numpy(self.b1.value),
            requires_grad=True
        )
        torch_layer.fc2.weight.data = torch.tensor(
            nntc.to_numpy(self.w2.value).T,  # Transpose back
            requires_grad=True
        )
        torch_layer.fc2.bias.data = torch.tensor(
            nntc.to_numpy(self.b2.value),
            requires_grad=True
        )
        
        return torch_layer
    
     def to_torch_with_grads(self) -> AdaptiveLayerTorch:
        torch_layer = self.to_torch()
        
        # Set gradients (transpose weight gradients back to PyTorch format)
        torch_layer.fc1.weight.grad = torch.tensor(
            nntc.to_numpy(self.w1.grad)  # Transpose back
        )
        torch_layer.fc1.bias.grad = torch.tensor(
            nntc.to_numpy(self.b1.grad)
        )
        torch_layer.fc2.weight.grad = torch.tensor(
            nntc.to_numpy(self.w2.grad)  # Transpose back
        )
        torch_layer.fc2.bias.grad = torch.tensor(
            nntc.to_numpy(self.b2.grad)
        )
        
        return torch_layer
