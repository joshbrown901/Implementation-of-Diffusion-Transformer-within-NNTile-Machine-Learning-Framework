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
- 🔜 Conduct **scalability testing** to support at least 4× model growth on equivalent hardware
- 🎯 Final goal: minimize **FID score** and enhance generative image quality

---

## 📊 Benchmarks

Test scripts are available for both implementations — those using constructors without random generators and those with them.

We compare NNTile-based Diffusion Transformer (DiT) blocks against Hugging Face’s PyTorch-based DiT using:
- ✅ Forward and backward **relative error**
- ✅ **Accuracy alignment** checks
- 🔍 Component-wise comparisons (e.g., MLP, MHSA, AdaLN-Zero)

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
