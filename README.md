# Implementation-of-Diffusion-Transformer-within-NNTile-Machine-Learning-Framework

## 📅 Roadmap

Efficient CPU-based Diffusion Transformer in NNtile (on STARPU), optimized for scalable, high-fidelity generative modeling. Benchmarked against PyTorch and Hugging Face. Roadmap includes training and generation pipelines, plus scalability testing for larger models and datasets.

The research is on-going and currently, we have implemented the components of Diffusion Transformer within NNTile using the Python-wrapper constructors without random generators and subsequently with random-generators.

#### 📊 Benchmarks

Tests scripts exists for both the implementations using the Python-wrapper constructors without random generators and with random-generators.

WE COMPARE the DiT-Block implementations for Hugging-Face Pytorch againsts NNTile implementations based on relative error and accuracy.

Further steps includes scalability tests, writing generation pipeline and training pipeline with the goal of scaling diffusion transformer so as to reduce the FID score and improve image generation quality.

## 📖 Citation

@misc {nntile2023,
  author       = {Aleksandr Mikhalev , Aleksandr Katrutsa , Konstantin Sozykin , Gleb Karpov , Daniel Bershatsky}, 
  title        = {NNTile: Task-based Machine Learning Framework},
  year         = {2023},
  howpublished = {\url{https://github.com/nntile/nntile}},
  note         = {Accessed: 2025-04-21}
}

@misc {huggingface2024,
  author       = {Hugging Face},
  title        = {Diffusers Library: DiT Transformer 2D Model},
  year         = {2024},
  howpublished = {\url{https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/models/transformers/dit_transformer_2d.py}},
  note         = {Accessed: 2025-04-21}
}

## 📬 Contact

Email: udobangjoshua@gmail.com

LinkedIn: https://www.linkedin.com/in/joshua-udobang-a852b1129?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app

For questions, feel free to reach out via GitHub Issues or open a discussion in the repo.
