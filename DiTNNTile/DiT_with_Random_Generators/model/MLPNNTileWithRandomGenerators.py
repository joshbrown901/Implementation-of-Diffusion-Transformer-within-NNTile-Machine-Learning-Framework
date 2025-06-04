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


class mlp(BaseLayer):
    emb:TensorMoments 
    w0: TensorMoments 
    w1: TensorMoments 
    w2: TensorMoments 
    w3: TensorMoments 
    w4: TensorMoments 
    w5: TensorMoments 
    
    b0: TensorMoments
    b1: TensorMoments 
    b2: TensorMoments 
    b3: TensorMoments 
    b4: TensorMoments 
    b5: TensorMoments 
    
    shift_msa: TensorMoments 
    scale_msa: TensorMoments 
    gate_msa: TensorMoments 
    shift_mlp: TensorMoments 
    scale_mlp: TensorMoments 
    gate_mlp: TensorMoments 
    emb_silu: Tensor 
    
    grad_activated0: Tensor
    grad_activated1: Tensor
    grad_activated2: Tensor
    grad_activated3: Tensor
    grad_activated4: Tensor
    grad_activated5: Tensor
    grad_activated: Tensor 
    
    def __init__(self, 
        emb:TensorMoments, 
        w0: TensorMoments, 
        w1: TensorMoments, 
        w2: TensorMoments, 
        w3: TensorMoments, 
        w4: TensorMoments, 
        w5: TensorMoments, 
        
        b0: TensorMoments,
        b1: TensorMoments, 
        b2: TensorMoments, 
        b3: TensorMoments, 
        b4: TensorMoments, 
        b5: TensorMoments, 
        
        shift_msa: TensorMoments, 
        scale_msa: TensorMoments, 
        gate_msa: TensorMoments, 
        shift_mlp: TensorMoments, 
        scale_mlp: TensorMoments, 
        gate_mlp: TensorMoments, 
        emb_silu: Tensor, 
        
        grad_activated0: Tensor,
        grad_activated1: Tensor,
        grad_activated2: Tensor,
        grad_activated3: Tensor,
        grad_activated4: Tensor,
        grad_activated5: Tensor,
        grad_activated: Tensor
    ):  
        
        self.emb = emb
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5
        
        self.shift_msa = shift_msa
        self.scale_msa = scale_msa
        self.gate_msa = gate_msa
        self.shift_mlp = shift_mlp
        self.scale_mlp = scale_mlp
        self.gate_mlp = gate_mlp
        
        self.emb_silu = emb_silu
        self.grad_activated0 = grad_activated0
        self.grad_activated1 = grad_activated1
        self.grad_activated2 = grad_activated2
        self.grad_activated3 = grad_activated3
        self.grad_activated4 = grad_activated4
        self.grad_activated5 = grad_activated5
        self.grad_activated = grad_activated
        
        super().__init__(
            [emb, w0, w1, w2, w3, w4, w5, b0, b1, b2, b3, b4, b5], 
            [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp], 
            [], 
            [emb_silu, grad_activated0, grad_activated1, grad_activated2, grad_activated3, grad_activated4, grad_activated5, grad_activated]
        )

    @staticmethod
    def generate_simple(
        emb: TensorMoments, 
        w0: TensorMoments,
        w1: TensorMoments,
        w2: TensorMoments,
        w3: TensorMoments,
        w4: TensorMoments,
        w5: TensorMoments,
        
        b0: TensorMoments, 
        b1: TensorMoments, 
        b2: TensorMoments, 
        b3: TensorMoments, 
        b4: TensorMoments, 
        b5: TensorMoments, 
        emb_traits: TensorTraits, 
        emb_distr,
        next_tag:int
    ):

        [B, D] = emb.value.shape
        [B_tile, D_tile] = emb.value.basetile_shape        
        
        emb_traits = TensorTraits([B, D], [B_tile, D_tile])
        emb_distr = emb.value.distribution

        w0_traits = TensorTraits([D, D], [D_tile, D_tile])
        w0_distr = w0.value.distribution

        w1_traits = TensorTraits([D, D], [D_tile, D_tile])
        w1_distr = w1.value.distribution

        w2_traits = TensorTraits([D, D], [D_tile, D_tile])
        w2_distr = w2.value.distribution

        w3_traits = TensorTraits([D, D], [D_tile, D_tile])
        w3_distr = w3.value.distribution

        w4_traits = TensorTraits([D, D], [D_tile, D_tile])
        w4_distr = w4.value.distribution

        w5_traits = TensorTraits([D, D], [D_tile, D_tile])
        w5_distr = w5.value.distribution

        b0_traits = TensorTraits([D], [D_tile])
        b0_distr = b0.value.distribution
        
        b1_traits = TensorTraits([D], [D_tile])
        b1_distr = b1.value.distribution
            
        b2_traits = TensorTraits([D], [D_tile])
        b2_distr = b2.value.distribution
            
        b3_traits = TensorTraits([D], [D_tile])
        b3_distr = b3.value.distribution
            
        b4_traits = TensorTraits([D], [D_tile])
        b4_distr = b4.value.distribution
            
        b5_traits = TensorTraits([D], [D_tile])
        b5_distr = b5.value.distribution
     

        shift_msa_traits = TensorTraits([B, D], [B_tile, D_tile])
        shift_msa_distr = [0] * shift_msa_traits.grid.nelems
        shift_msa_value = type(emb.value)(shift_msa_traits, shift_msa_distr, next_tag)
        next_tag = shift_msa_value.next_tag
        shift_msa_grad = type(emb.value)(shift_msa_traits, shift_msa_distr, next_tag)
        next_tag = shift_msa_grad.next_tag
        shift_msa = TensorMoments(shift_msa_value, shift_msa_grad, True)

        scale_msa_traits = TensorTraits([B, D], [B_tile, D_tile])
        scale_msa_distr = [0] * scale_msa_traits.grid.nelems
        scale_msa_value = type(emb.value)(scale_msa_traits, scale_msa_distr, next_tag)
        next_tag = scale_msa_value.next_tag
        scale_msa_grad = type(emb.value)(scale_msa_traits, scale_msa_distr, next_tag)
        next_tag = scale_msa_grad.next_tag
        scale_msa = TensorMoments(scale_msa_value, scale_msa_grad, True)

        gate_msa_traits = TensorTraits([B, D], [B_tile, D_tile])
        gate_msa_distr = [0] * gate_msa_traits.grid.nelems
        gate_msa_value = type(emb.value)(gate_msa_traits, gate_msa_distr, next_tag)
        next_tag = gate_msa_value.next_tag
        gate_msa_grad = type(emb.value)(gate_msa_traits, gate_msa_distr, next_tag)
        next_tag = gate_msa_grad.next_tag
        gate_msa = TensorMoments(gate_msa_value, gate_msa_grad, True)

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

        gate_mlp_traits = TensorTraits([B, D], [B_tile, D_tile])
        gate_mlp_distr = [0] * gate_mlp_traits.grid.nelems
        gate_mlp_value = type(emb.value)(gate_mlp_traits, gate_mlp_distr, next_tag)
        next_tag = gate_mlp_value.next_tag
        gate_mlp_grad = type(emb.value)(gate_mlp_traits, gate_mlp_distr, next_tag)
        next_tag = gate_mlp_grad.next_tag
        gate_mlp = TensorMoments(gate_mlp_value, gate_mlp_grad, True)

        emb_silu = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = emb_silu.next_tag


        grad_activated0 = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = grad_activated0.next_tag

        grad_activated1 = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = grad_activated1.next_tag

        grad_activated2 = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = grad_activated2.next_tag

        grad_activated3 = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = grad_activated3.next_tag

        grad_activated4 = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = grad_activated4.next_tag

        grad_activated5 = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = grad_activated5.next_tag

        grad_activated = type(emb.value)(emb_traits, emb_distr, next_tag)
        next_tag = grad_activated.next_tag

        layer = mlp(
            emb, 
            w0, 
            w1, 
            w2, 
            w3, 
            w4, 
            w5, 
            b0, 
            b1, 
            b2, 
            b3, 
            b4, 
            b5, 
            shift_msa, 
            scale_msa, 
            gate_msa, 
            shift_mlp, 
            scale_mlp, 
            gate_mlp, 
            emb_silu, 
            grad_activated0, 
            grad_activated1, 
            grad_activated2, 
            grad_activated3, 
            grad_activated4, 
            grad_activated5, 
            grad_activated
        ) 
        
        return (layer, next_tag)

    def forward_async(self):
        nntf.silu_forward_async(self.emb.value, self.emb_silu)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.w0.value, 0.0, self.shift_msa.value, 1, 0, 0) 
        nntf.add_slice_inplace_async(1.0, self.b0.value, 1.0, self.shift_msa.value, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.w1.value, 0.0, self.scale_msa.value, 1, 0, 0) 
        nntf.add_slice_inplace_async(1.0, self.b1.value, 1.0, self.scale_msa.value, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.w2.value, 0.0, self.gate_msa.value, 1, 0, 0) 
        nntf.add_slice_inplace_async(1.0, self.b2.value, 1.0, self.gate_msa.value, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.w3.value, 0.0, self.shift_mlp.value, 1, 0, 0) 
        nntf.add_slice_inplace_async(1.0, self.b3.value, 1.0, self.shift_mlp.value, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.w4.value, 0.0, self.scale_mlp.value, 1, 0, 0) 
        nntf.add_slice_inplace_async(1.0, self.b4.value, 1.0, self.scale_mlp.value, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.emb_silu, nntile.nntile_core.trans, self.w5.value, 0.0, self.gate_mlp.value, 1, 0, 0) 
        nntf.add_slice_inplace_async(1.0, self.b5.value, 1.0, self.gate_mlp.value, 0)
        
    def backward_async(self):
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.shift_msa.grad, nntile.nntile_core.notrans, self.w0.value, 0.0, self.grad_activated0, 1, 0, 0) 
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.scale_msa.grad, nntile.nntile_core.notrans, self.w1.value, 0.0, self.grad_activated1, 1, 0, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.gate_msa.grad, nntile.nntile_core.notrans, self.w2.value, 0.0, self.grad_activated2, 1, 0, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.shift_mlp.grad, nntile.nntile_core.notrans, self.w3.value, 0.0, self.grad_activated3, 1, 0, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.scale_mlp.grad, nntile.nntile_core.notrans, self.w4.value, 0.0, self.grad_activated4, 1, 0, 0) 
        nntf.gemm_async(1.0, nntile.nntile_core.notrans, self.gate_mlp.grad, nntile.nntile_core.notrans, self.w5.value, 0.0, self.grad_activated5, 1, 0, 0)
        
        nntf.fill_async(0.0, self.grad_activated)
        
        nntf.add_inplace_async(1.0, self.grad_activated0, 1.0, self.grad_activated)
        nntf.add_inplace_async(1.0, self.grad_activated1, 1.0, self.grad_activated)
        nntf.add_inplace_async(1.0, self.grad_activated2, 1.0, self.grad_activated)
        nntf.add_inplace_async(1.0, self.grad_activated3, 1.0, self.grad_activated)
        nntf.add_inplace_async(1.0, self.grad_activated4, 1.0, self.grad_activated)
        nntf.add_inplace_async(1.0, self.grad_activated5, 1.0, self.grad_activated)
        
        nntf.silu_backward_async(self.emb.value, self.grad_activated, self.emb.grad)

        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.shift_msa.grad, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.w0.grad, 1, 0, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.scale_msa.grad, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.w1.grad, 1, 0, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.gate_msa.grad, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.w2.grad, 1, 0, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.shift_mlp.grad, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.w3.grad, 1, 0, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.scale_mlp.grad, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.w4.grad, 1, 0, 0)
        nntf.gemm_async(1.0, nntile.nntile_core.trans, self.gate_mlp.grad, nntile.nntile_core.notrans, self.emb_silu, 0.0, self.w5.grad, 1, 0, 0)

       
        nntf.sum_slice_async(1.0, self.shift_msa.grad, 0.0, self.b0.grad, 0)
        nntf.sum_slice_async(1.0, self.scale_msa.grad, 0.0, self.b1.grad, 0)
        nntf.sum_slice_async(1.0, self.gate_msa.grad, 0.0, self.b2.grad, 0)
        nntf.sum_slice_async(1.0, self.shift_mlp.grad, 0.0, self.b3.grad, 0)
        nntf.sum_slice_async(1.0, self.scale_mlp.grad, 0.0, self.b4.grad, 0)
        nntf.sum_slice_async(1.0, self.gate_mlp.grad, 0.0, self.b5.grad, 0)

    @classmethod
    def from_torch(cls, torch_module: MLPTorch,
                   emb: TensorMoments,
                   next_tag: int):

        # 1) pull out the big W and b
        W_full = torch_module.W.data.cpu().numpy()
        b_full = torch_module.b.data.cpu().numpy() 

        # 2) split into six sub-blocks
        D = W_full.shape[1]
        Ws = [W_full[i*D:(i+1)*D] for i in range(6)]
        bs = [b_full[i*D:(i+1)*D] for i in range(6)]

        # 3) build TensorMoments for each Wi, bi
        w_tms = []
        for W_np in Ws:
            W_val  = nntc.from_array(W_np)
            W_grad = nntc.zeros_like(W_val)
            w_tms.append(TensorMoments(W_val, W_grad, True))
        b_tms = []
        for b_np in bs:
            if b_np is not None:
                b_val  = nntc.from_array(b_np)
                b_grad = nntc.zeros_like(b_val)
                b_tms.append(TensorMoments(b_val, b_grad, True))
            else:
                # no bias for this block
                b_tms.append(TensorMoments(None, None, False))

        # 4) capture emb traits/distribution
        emb_traits = TensorTraits(emb.value.shape, emb.value.basetile_shape)
        emb_distr  = emb.value.distribution

        # 5) delegate to generate_simple to wire all internals
        layer, next_tag = cls.generate_simple(
            emb,
            w_tms[0], w_tms[1], w_tms[2],
            w_tms[3], w_tms[4], w_tms[5],
            b_tms[0], b_tms[1], b_tms[2],
            b_tms[3], b_tms[4], b_tms[5],
            emb_traits, emb_distr, next_tag
        )
        return layer, next_tag

    def to_torch(self) -> MLPTorch:

        Ws = [ nntc.to_numpy(getattr(self, f"w{i}").value) for i in range(6) ] 
        W_full = np.vstack(Ws)                                                 

        # 2) same for biases
        if self.b0 is not None:
            bs = [ nntc.to_numpy(getattr(self, f"b{i}").value) for i in range(6) ]
            b_full = np.hstack(bs)
        else:
            b_full = None

        # 3) create a new MLPTorch and copy
        torch_mod = MLPTorch(self.emb.value.shape[-1], bias=(b_full is not None))
        with torch.no_grad():
            torch_mod.W.copy_(torch.from_numpy(W_full))
            if b_full is not None:
                torch_mod.b.copy_(torch.from_numpy(b_full))
        return torch_mod

    def to_torch_with_grads(self) -> MLPTorch:
        torch_mod = self.to_torch()

        # 1) gather W grads
        Wg_blocks = [ nntc.to_numpy(getattr(self, f"w{i}").grad) for i in range(6) ]
        Wg_full   = np.vstack(Wg_blocks)
        torch_mod.W.grad = torch.from_numpy(Wg_full).clone().to(torch.float32)

        # 2) gather bias grads if present
        if self.b0 is not None:
            bg_blocks = [ nntc.to_numpy(getattr(self, f"b{i}").grad) for i in range(6) ]
            bg_full   = np.hstack(bg_blocks)
            torch_mod.b.grad = torch.from_numpy(bg_full).clone().to(torch.float32)

        return torch_mod