import nntile
import numpy as np
nntile_config = nntile.starpu.Config(1,0,0,0)
nntile.starpu.init()
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, TensorTraits


class FeedForward(BaseLayer):
    x: TensorMoments 
    w1: TensorMoments 
    b1: TensorMoments 
    w2: TensorMoments 
    b2: TensorMoments  
    x11: Tensor
    x1: TensorMoments
    a1: TensorMoments
    out1: Tensor
    out: TensorMoments
    grad_out1: Tensor
    grad_a1: Tensor
    grad_x1: Tensor
    x_flat: Tensor
    
    def __init__(self,
                 x: TensorMoments, 
                 w1: TensorMoments, 
                 b1: TensorMoments,                  
                 w2: TensorMoments, 
                 b2: TensorMoments, 
                 x11: Tensor, 
                 x1: TensorMoments,
                 a1: TensorMoments,
                 out1: Tensor,
                 out: TensorMoments,                  
                 grad_out1: Tensor,
                 grad_a1: Tensor,
                 grad_x1: Tensor,
                 x_flat: Tensor):
        
        self.x = x
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.x11 = x11
        self.x1 = x1
        self.a1 = a1
        self.out1 = out1
        self.out = out        
        self.grad_out1 = grad_out1
        self.grad_a1 = grad_a1
        self.grad_x1 = grad_x1
        self.x_flat = x_flat
        
        super().__init__([x, w1, b1, w2, b2], [out], [], [x11, x1, a1, out1, grad_out1, grad_a1, grad_x1, x_flat])
    @staticmethod
    def generate_simple(x: TensorMoments, 
                        w1: TensorMoments,
                        b1: TensorMoments,                      
                        w2: TensorMoments, 
                        b2: TensorMoments,
                        out: TensorMoments,
                        x_traits: TensorTraits, 
                        x_distr, next_tag:int):

        [B, N, D] = x.value.shape
        [B_tile, N_tile, D_tile] = x.value.basetile_shape

        [F] = b1.value.shape
        [F_tile] = b1.value.basetile_shape
           
        x_traits = TensorTraits([B, N, D], [B_tile, N_tile, D_tile])
        x_distr = x.value.distribution
        
        w1_traits = TensorTraits([F, D], [F_tile, D_tile])
        w1_distr = w1.value.distribution
        
        w2_traits = TensorTraits([D, F], [D_tile, F_tile])
        w2_distr = w2.value.distribution
        
        b1_traits = TensorTraits([F], [F_tile])
        b1_distr = b1.value.distribution
        
        b2_traits = TensorTraits([D], [D_tile])
        b2_distr = b2.value.distribution
        
        out_traits = TensorTraits([B, N, D], [B_tile, N_tile, D_tile])
        out_distr = [0] * out_traits.grid.nelems
        out_value = type(x.value)(out_traits, out_distr, next_tag)
        next_tag = out_value.next_tag
        out_grad = type(x.value)(out_traits, out_distr, next_tag)
        next_tag = out_grad.next_tag
        out = TensorMoments(out_value, out_grad, True)

        x11_shape     = [B, N, F]
        x11_basetile  = [B_tile, N_tile, F_tile]
        x11_traits = TensorTraits(x11_shape, x11_basetile)
        x11_distr = [0] * x11_traits.grid.nelems
        x11 = type(x.value)(x11_traits, x11_distr, next_tag)
        next_tag = x11.next_tag

        x1_shape     = [B, N, F]
        x1_basetile_shape  = [B_tile, N_tile, F_tile]
        x1_traits = TensorTraits(x1_shape, x1_basetile_shape)
        x1_distr = [0] * x1_traits.grid.nelems
        x1_value = type(x.value)(x1_traits, x1_distr, next_tag)
        next_tag = x1_value.next_tag
        x1_grad = type(x.value)(x1_traits, x1_distr, next_tag)
        next_tag = x1_grad.next_tag
        x1 = TensorMoments(x1_value, x1_grad, True)

        a1_shape     = [B, N, F]
        a1_basetile_shape  = [B_tile, N_tile, F_tile]
        a1_traits = TensorTraits(a1_shape, a1_basetile_shape)
        a1_distr = [0] * a1_traits.grid.nelems
        a1_value = type(x.value)(a1_traits, a1_distr, next_tag)
        next_tag = a1_value.next_tag
        a1_grad = type(x.value)(a1_traits, a1_distr, next_tag)
        next_tag = a1_grad.next_tag
        a1 = TensorMoments(a1_value, a1_grad, True)

        out1 = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = out1.next_tag

        grad_out1_shape = [B*N, D]
        grad_out1_basetile_shape  = [B_tile*N_tile, D_tile]
        grad_out1_traits = TensorTraits(grad_out1_shape, grad_out1_basetile_shape)
        grad_out1_distr = [0] * grad_out1_traits.grid.nelems
        grad_out1 = type(x.value)(grad_out1_traits, grad_out1_distr, next_tag)
        next_tag = grad_out1.next_tag

        grad_a1_shape = [B*N, F]
        grad_a1_basetile_shape  = [B_tile*N_tile, F_tile]
        grad_a1_traits = TensorTraits(grad_a1_shape, grad_a1_basetile_shape)
        grad_a1_distr = [0] * grad_a1_traits.grid.nelems
        grad_a1 = type(x.value)(grad_a1_traits, grad_a1_distr, next_tag)
        next_tag = grad_a1.next_tag

        grad_x1_shape = [B*N, F]
        grad_x1_basetile_shape  = [B_tile*N_tile, F_tile]
        grad_x1_traits = TensorTraits(grad_x1_shape, grad_x1_basetile_shape)
        grad_x1_distr = [0] * grad_x1_traits.grid.nelems
        grad_x1 = type(x.value)(grad_x1_traits, grad_x1_distr, next_tag)
        next_tag = grad_x1.next_tag

        x_flat_shape = [B*N, D]
        x_flat_basetile_shape  = [B_tile*N_tile, D_tile]
        x_flat_traits = TensorTraits(x_flat_shape, x_flat_basetile_shape)
        x_flat_distr = [0] * x_flat_traits.grid.nelems
        x_flat = type(x.value)(x_flat_traits, x_flat_distr, next_tag)
        next_tag = x_flat.next_tag

        return FeedForward(x, w1, b1, w2, b2, x11, x1, a1, out1, out, grad_out1, grad_a1, grad_x1, x_flat), next_tag

    def forward_async(self):
        [self.B, self.N, self.D] = self.x.value.shape
        [self.F] = self.b1.value.shape
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.x.value, nntile.nntile_core.trans, self.w1.value, 0.0, self.x11, 1, 0, 0)
        self.x.value.wont_use()
        self.w1.value.wont_use()       
        nntf.add_fiber_async(1.0, self.b1.value, 1.0, self.x11, self.x1.value, 2, 0)
        self.b1.value.wont_use()
        self.x11.wont_use()       
        nntf.gelutanh_async(self.x1.value, self.a1.value)
        self.x1.value.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.a1.value, nntile.nntile_core.trans, self.w2.value, 0.0, self.out1, 1, 0, 0)
        self.a1.value.wont_use()
        self.w2.value.wont_use()        
        nntf.add_fiber_async(1.0, self.b2.value, 1.0, self.out1, self.out.value, 2, 0)
        self.b2.value.wont_use()
        self.out1.wont_use()
        self.out.value.wont_use()

    def backward_async(self):
        self.grad_out_np = nntc.to_numpy(self.out.grad).reshape(-1, self.D)
        self.grad_out1 = nntc.from_array(self.grad_out_np)
        nntf.gelutanh_async(self.x1.value, self.a1.value)
        self.grad_a1_np = nntc.to_numpy(self.a1.value).reshape(-1, self.F)
        self.grad_a1 = nntc.from_array(self.grad_a1_np)
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_out1, nntile.nntile_core.notrans, self.grad_a1, 0.0, self.w2.grad, 1, 0, 0)        
        #return self.w2.grad        
        self.grad_out1.wont_use()
        self.grad_a1.wont_use()
        self.a1.value.wont_use()
        self.w2.grad.wont_use()        
        nntf.sum_fiber_async(1.0, self.out.grad, 0.0, self.b2.grad, 2, 0)
        self.b2.grad.wont_use()
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.out.grad, nntile.nntile_core.notrans, self.w2.value, 0.0, self.a1.grad, 1, 0, 0)            
        self.out.grad.wont_use()        
        self.w2.value.wont_use()        
        nntf.gelutanh_backward_async(self.x1.value, self.a1.grad, self.x1.grad)
        self.a1.grad.wont_use()
        self.grad_x1 = nntc.from_array(nntc.to_numpy(self.x1.grad).reshape(-1, self.F))
        self.x_flat = nntc.from_array(self.x.value.reshape(-1, self.D))   
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.grad_x1, nntile.nntile_core.notrans, self.x_flat, 0.0, self.w1.grad, 1, 0, 0)        
        self.grad_x1.wont_use()
        self.x_flat.wont_use() 
        self.w1.grad.wont_use()
        nntf.sum_fiber_async(1.0, self.x1.grad, 0.0, self.b1.grad, 2, 0)   
        self.x1.grad.wont_use()
        self.b1.grad.wont_use()       
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.x1.grad, nntile.nntile_core.notrans, self.w1.value, 0.0, self.x.grad, 1, 0, 0)
        self.w1.value.wont_use()
        self.x.grad.wont_use()
    
    @classmethod
    def from_torch(
        cls,
        torch_module: FeedForwardTorch,
        x: TensorMoments,
        next_tag: int
    ):

        W1_np = torch_module.fc1.weight.data.cpu().numpy()
        b1_np = torch_module.fc1.bias.data.cpu().numpy()
        W2_np = torch_module.fc2.weight.data.cpu().numpy()
        b2_np = torch_module.fc2.bias.data.cpu().numpy()

        w1_val  = nntc.from_array(W1_np)
        w1_grad = nntc.zeros_like(w1_val)
        w1_tm   = TensorMoments(w1_val, w1_grad, True)

        b1_val  = nntc.from_array(b1_np)
        b1_grad = nntc.zeros_like(b1_val)
        b1_tm   = TensorMoments(b1_val, b1_grad, True)

        w2_val  = nntc.from_array(W2_np)
        w2_grad = nntc.zeros_like(w2_val)
        w2_tm   = TensorMoments(w2_val, w2_grad, True)

        b2_val  = nntc.from_array(b2_np)
        b2_grad = nntc.zeros_like(b2_val)
        b2_tm   = TensorMoments(b2_val, b2_grad, True)

        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        x_distr  = x.value.distribution

        out_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        out_distr = [0] * out_traits.grid.nelems
        out_value = type(x.value)(out_traits, out_distr, next_tag)
        next_tag = out_value.next_tag
        out_grad = type(x.value)(out_traits, out_distr, next_tag)
        next_tag = out_grad.next_tag
        out = TensorMoments(out_value, out_grad, True)
        
        layer, next_tag = cls.generate_simple(
            x, w1_tm, b1_tm, w2_tm, b2_tm, out,
            x_traits, x_distr, next_tag)
        return layer, next_tag

    def to_torch(self) -> FeedForwardTorch:
        hidden_size = self.x.value.shape[-1]
        ff_hidden   = self.w2.value.shape[-1]
        torch_mod   = FeedForwardTorch(hidden_size, ff_hidden)

        W1_arr = nntc.to_numpy(self.w1.value)
        b1_arr = nntc.to_numpy(self.b1.value)
        W2_arr = nntc.to_numpy(self.w2.value)
        b2_arr = nntc.to_numpy(self.b2.value)

        with torch.no_grad():
            torch_mod.fc1.weight.copy_(torch.from_numpy(W1_arr))
            torch_mod.fc1.bias.copy_(  torch.from_numpy(b1_arr))
            torch_mod.fc2.weight.copy_(torch.from_numpy(W2_arr))
            torch_mod.fc2.bias.copy_(  torch.from_numpy(b2_arr))

        return torch_mod

    def to_torch_with_grads(self) -> FeedForwardTorch:
        torch_mod = self.to_torch()

        W1g = nntc.to_numpy(self.w1.grad)
        b1g = nntc.to_numpy(self.b1.grad)
        W2g = nntc.to_numpy(self.w2.grad)
        b2g = nntc.to_numpy(self.b2.grad)

        torch_mod.fc1.weight.grad = torch.from_numpy(W1g).clone()
        torch_mod.fc1.bias.  grad = torch.from_numpy(b1g).clone()
        torch_mod.fc2.weight.grad = torch.from_numpy(W2g).clone()
        torch_mod.fc2.bias.  grad = torch.from_numpy(b2g).clone()

        return torch_mod