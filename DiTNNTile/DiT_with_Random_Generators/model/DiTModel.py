class DiTBlock(BaseModel):
    next_tag:int
    #patchify: PatchEmbedding
    #conditioning: CombinedTimeStepsLabelEmbeddings
    dit_mlp : mlp
    attn_norm: LayerNorm
    pre_attn_scale_shift: scale_shift
    attn_block: Attention
    post_attn_scale: scale_shift
    ff_norm: LayerNorm
    pre_ff_scale_shift: scale_shift
    FFN: FeedForward
    post_ff_scale: scale_shift
    adaptive_layer: AdaptiveLayer
    post_ff_norm: LayerNorm
    out_scale_shift: scale_shift
    proj_out: Linear
    #reshape: Unpatchify

    def __init__(self, 
                 x: TensorMoments, 
                 emb: TensorMoments,
                dit_mlp : mlp
                attn_norm: LayerNorm
                pre_attn_scale_shift: scale_shift
                attn_block: Attention
                post_attn_scale: scale_shift
                ff_norm: LayerNorm
                pre_ff_scale_shift: scale_shift
                FFN: FeedForward
                post_ff_scale: scale_shift
                adaptive_layer: AdaptiveLayer
                post_ff_norm: LayerNorm
                out_scale_shift: scale_shift
                proj_out: Linear
                config: DiT_config_NNTile):

        self.mlp = mlp
        layers = [dit_mlp, attn_norm, pre_attn_scale_shift, attn_block, post_attn_scale, ff_norm, pre_ff_scale_shift, FFN, post_ff_scale, adaptive_layer, post_ff_norm, out_scale_shift, proj_out]
        activations = [x] +  [emb] + mlp.activations_output + attn_norm.activations_output + pre_attn_scale_shift.activations_output + \
                      attn_block.activations_output + post_attn_scale.activations_output + ff_norm.activations_output + \
                      pre_ff_scale_shift.activations_output + FFN.activations_output + post_ff_scale.activations_output + \
                      adaptive_layer.activations_output + post_ff_norm.activations_output + out_scale_shift.activations_output + \
                      proj_out.activations_output
        self.config = config

        super().__init__(activations, layers, config)

    def forward_async (self,
                       x: TensorMoments,
                       emb: TensorMoments,
                       #kv_cache: Optional[KVCache] = None
                      ):
        
        mlp.forward.async(emb)
        attn_norm.forward_async(x)
        pre_attn_scale_shift.forward_async(attn_norm.y.value, mlp.scale_msa.value, mlp.shift_msa.value)
        attn_block.forward_async(pre_attn_scale_shift.y.value)
        post_attn_scale.forward_async(attn_block.y.value, mlp.gate_msa.value, x)
        ff_norm.forward_async(post_attn_scale.y.value)
        pre_ff_scale_shift.forward_async(ff_norm.y.value, mlp.scale_mlp.value, mlp.shift_mlp.value)
        FFN.forward_async(pre_ff_scale_shift.y.value)
        post_ff_scale.forward_async(FFN.y.value, mlp.gate_mlp.value, post_attn_scale.y.value)
        adaptive_layer.forward_async(emb)
        post_ff_norm.forward_async(post_ff_scale.y.value)
        out_scale_shift.forward_async(post_ff_norm.y.value, adaptive_layer.scale_mlp.value, adaptive_layer.ahift_mlp.value)
        proj_out.forward_async(out_scale_shift.y.value)

    def backward_async(self):
        self.proj_out.backward_async()
        self.out_scale_shift.backward_async(self.proj_out.y.grad, self.post_ff_norm.y.value, self.adaptive_layer.scale_mlp.value,
            self.adaptive_layer.shift_mlp.value)
  
        self.post_ff_norm.backward_async(self.out_scale_shift.x.grad)

        self.adaptive_layer.backward_async(self.adaptive_layer.scale_mlp.grad, self.adaptive_layer.shift_mlp.grad)

        self.post_ff_scale.backward_async(self.post_ff_norm.y.grad, self.FFN.y.value, self.mlp.gate_mlp.value,
            self.post_attn_scale.y.value)
 
        self.FFN.backward_async(self.post_ff_scale.y.grad)

        self.pre_ff_scale_shift.backward_async(self.FFN.y.grad, self.ff_norm.y.value, self.mlp.scale_mlp.value, self.mlp.shift_mlp.value)

 
        self.ff_norm.backward_async(self.pre_ff_scale_shift.y.grad)

        self.post_attn_scale.backward_async(self.ff_norm.y.grad, self.attn_block.y.value, self.mlp.gate_msa.value, self.x.value)

        self.attn_block.backward_async(self.post_attn_scale.y.grad)

        self.pre_attn_scale_shift.backward_async(self.attn_block.y.grad, self.attn_norm.y.value, self.mlp.scale_msa.value,
            self.mlp.shift_msa.value)

        self.attn_norm.backward_async(self.pre_attn_scale_shift.y.grad)

        self.mlp.backward_async(self.attn_norm.y.grad, self.emb.value)

    @classmethod
    def from_torch(cls,
                   torch_block: DiTBlockTorch,
                   x_tm: TensorMoments,
                   emb_tm: TensorMoments,
                   next_tag: int,
                   config: DiT_config_NNTile):

        dit_mlp_layer, next_tag    = mlp.from_torch(torch_block.dit_mlp, emb_tm, next_tag)
        attn_norm_layer            = LayerNorm.from_torch(torch_block.attn_norm)
        pre_attn_ss_layer          = scale_shift.from_torch(torch_block.pre_attn_scale_shift)
        attn_block_layer           = Attention.from_torch(torch_block.attn_block)
        post_attn_ss_layer         = scale_shift.from_torch(torch_block.post_attn_scale)
        ff_norm_layer              = LayerNorm.from_torch(torch_block.ff_norm)
        pre_ff_ss_layer            = scale_shift.from_torch(torch_block.pre_ff_scale_shift)
        ffn_layer, next_tag        = FeedForward.from_torch(torch_block.FFN, pre_ff_ss_layer.y, next_tag)
        post_ff_ss_layer           = scale_shift.from_torch(torch_block.post_ff_scale)
        adaptive_layer             = AdaptiveLayer.from_torch(torch_block.adaptive_layer, emb_tm, next_tag)
        post_ff_norm_layer         = LayerNorm.from_torch(torch_block.post_ff_norm)
        out_ss_layer               = scale_shift.from_torch(torch_block.out_scale_shift)
        proj_layer                 = Linear.from_torch(torch_block.proj_out)

        layer = cls(
            x_tm, emb_tm,
            dit_mlp_layer,
            attn_norm_layer,
            pre_attn_ss_layer,
            attn_block_layer,
            post_attn_ss_layer,
            ff_norm_layer,
            pre_ff_ss_layer,
            ffn_layer,
            post_ff_ss_layer,
            adaptive_layer,
            post_ff_norm_layer,
            out_ss_layer,
            proj_layer,
            config
        )
        return layer, next_tag

    def to_torch(self) -> DiTBlockTorch:
             
        torch_model = DiTModelTorch(self.config)
        torch_block = torch_model.layers[self.config.block_idx]
        torch_block.dit_mlp              = self.dit_mlp.to_torch()
        torch_block.attn_norm            = self.attn_norm.to_torch()
        torch_block.pre_attn_scale_shift = self.pre_attn_scale_shift.to_torch()
        torch_block.attn_block           = self.attn_block.to_torch()
        torch_block.post_attn_scale      = self.post_attn_scale.to_torch()
        torch_block.ff_norm              = self.ff_norm.to_torch()
        torch_block.pre_ff_scale_shift   = self.pre_ff_scale_shift.to_torch()
        torch_block.FFN                  = self.FFN.to_torch()
        torch_block.post_ff_scale        = self.post_ff_scale.to_torch()
        torch_block.adaptive_layer       = self.adaptive_layer.to_torch()
        torch_block.post_ff_norm         = self.post_ff_norm.to_torch()
        torch_block.out_scale_shift      = self.out_scale_shift.to_torch()
        torch_block.proj_out             = self.proj_out.to_torch()

        return torch_block

    def to_torch_with_grads(self) -> DiTBlockTorch:
             
        torch_block = self.to_torch()

        torch_block.dit_mlp              = self.dit_mlp.to_torch_with_grads()
        torch_block.attn_norm            = self.attn_norm.to_torch_with_grads()
        torch_block.pre_attn_scale_shift = self.pre_attn_scale_shift.to_torch_with_grads()
        torch_block.attn_block           = self.attn_block.to_torch_with_grads()
        torch_block.post_attn_scale      = self.post_attn_scale.to_torch_with_grads()
        torch_block.ff_norm              = self.ff_norm.to_torch_with_grads()
        torch_block.pre_ff_scale_shift   = self.pre_ff_scale_shift.to_torch_with_grads()
        torch_block.FFN                  = self.FFN.to_torch_with_grads()
        torch_block.post_ff_scale        = self.post_ff_scale.to_torch_with_grads()
        torch_block.adaptive_layer       = self.adaptive_layer.to_torch_with_grads()
        torch_block.post_ff_norm         = self.post_ff_norm.to_torch_with_grads()
        torch_block.out_scale_shift      = self.out_scale_shift.to_torch_with_grads()
        torch_block.proj_out             = self.proj_out.to_torch_with_grads()

        return torch_block   
