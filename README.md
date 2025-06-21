# Implementation of Diffusion Transformer within NNTile Machine Learning Framework

Efficient CPU-based Diffusion Transformer implemented in the NNtile framework (built on STARPU), optimized for scalable, high-fidelity generative modeling. Benchmarked against PyTorch and Hugging Face implementations. The roadmap includes a modular training pipeline, a flexible generation pipeline, and scalability testing for larger models and datasets.

---

## Setup
Run the code on NNTile Docker Image below
```bash
docker pull ghcr.io/nntile/nntile:1.1.0-starpu1.4.7-cuda12.4.0-ubuntu22.04
```
## 📅 Roadmap

This research is ongoing. Currently, we have implemented the core components of the Diffusion Transformer within NNTile using Python wrapper constructors — both **without random generators** and **with random generators**.

### Next milestones:
- ✅ Validate DiT-Block components against Hugging Face’s PyTorch reference using relative error metrics
- 🔜 Develop and integrate the **generation pipeline**
- 🔜 Build a robust **training pipeline**
- ✅ Conduct **scalability testing** to support at least 4× model growth on equivalent hardware
- 🎯 Final goal: minimize **FID score** and enhance generative image quality

---

## 📊 Benchmarks
### Results Achieved
### Scale and Shift

Forward Relative Error: 0.0 

Grad X Relative Error: 0.0 

Grad Scale Relative Error: 2.7376e-08

Grad Shift Relative Error: 0.0 

### Scale and Skip-Connection

Forward Relative Error: 0.0 

Grad X Relative Error: 0.0 

Grad Scale Relative Error: 2.7376e-08

Grad Shift Relative Error: 0.0 

### MLP

shift_msa: NNTile vs Torch error: 3.14888e-08

scale_msa: NNTile vs Torch error: 1.377236e-08

gate_msa: NNTile vs Torch error: 3.7221852e-08

shift_mlp: NNTile vs Torch error: 1.4630345e-08

scale_mlp: NNTile vs Torch error: 1.342537e-08

gate_mlp: NNTile vs Torch error: 5.81201e-08

Grad emb rel error (NNTile vs PyTorch): 3.002553e-08

Grad W rel error (NNTile vs PyTorch): 1.7904173e-08

Grad b rel error (NNTile vs Pytorch): 1.8100875e-08

### Pointwise FeedForward Neural Network

Relative errors:

Forward Backward:

out: NNTile vs Torch: 5.43e-06 

dx: NNTile vs Torch: 8.62e-06

dw1: NNTile vs Torch: 3.40e-06 

dw2: NNTile vs Torch: 8.17e-06

db1: NNTile vs Torch: 4.97e-07

db2: NNTile vs Torch: 1.45e-07

### MultiHeadSelfAttention

out: 9.602e-08

input grad: 1.44e-07

qkv weight: 1.30e-07

qkv bias: 8.07e-08

out weight: 1.522e-07

out bias: 3.974e-08

### AdaptiveLayerNormZero

Relative errors between Torch and NNTile implementations

x out: 1.65e-05

gate msa: 9.01e-08

shift mlp: 0.00e+00

scale mlp: 1.09e-07

gate mlp: 4.91e-06

d x: 2.39e-06

d emb: 2.39e-06



Test scripts are available for both implementations — those using constructors without random generators and those with them.

We compare NNTile-based Diffusion Transformer (DiT) blocks against Hugging Face’s PyTorch-based DiT using:
- ✅ Forward and backward **relative error**
- ✅ **Accuracy alignment** checks
- 🔍 Component-wise comparisons (e.g., MLP, MHSA, AdaLN-Zero)
- ✅ Scalability Tests

Future benchmark directions:
- Performance profiling
- FID tracking on standard datasets
- Scaling trends

---

## 📖 Citation

@misc{nntile2023,
  author       = {Aleksandr Mikhalev, Aleksandr Katrutsa, Konstantin Sozykin, Gleb Karpov, Daniel Bershatsky},
  title        = {NNTile: Task-based Machine Learning Framework},
  year         = {2023},
  howpublished = {\url{https://github.com/nntile/nntile}},
  note         = {Accessed: 2025-04-21}
}

@misc{huggingface2024,
  author       = {Hugging Face},
  title        = {Diffusers Library: DiT Transformer 2D Model},
  year         = {2024},
  howpublished = {\url{https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/models/transformers/dit_transformer_2d.py}},
  note         = {Accessed: 2025-04-21}
}

## 📬 Contact

📧 Email: udobangjoshua@gmail.com

🔗 LinkedIn: Joshua Udobang

For questions, feel free to reach out via GitHub Issues or open a discussion in the repo.

## 🙏 Acknowledgement

I would like to thank Professor Aleksandr Mikhalev for the opportunity to be a part of this ongoing research, and for the guidance and support in contributing to and utilizing the task-based parallelism paradigm NNTile.
