
# DiffWave: A Versatile Diffusion Probabilistic Model(Implementation)

A PyTorch implementation of the **DiffWave** architecture for non-autoregressive waveform generation. This project converts white noise into structured audio through a Markov chain of  refinement steps.

## Core Technical Features

* **Non-Autoregressive:** Parallel waveform synthesis via bidirectional dilated convolutions.
* **Diffusion Process:** Implements the Ho et al. (2020) parameterization for  and .
* **Conditioning:** Supports both Local (Mel Spectrogram) and Global (Discrete Labels) conditioning.
* **Efficiency:** Receptive field expansion across  iterations allows for high-fidelity audio with a small parameter footprint (2.64M parameters).

## Repository Structure

* `/model`: Core PyTorch modules, including the bidirectional Dilated Conv stack and diffusion-step embeddings.
* `/scripts`: Training and inference scripts for the LJ Speech and SC09 datasets.
* `/notebooks`: A Jupyter implementation for rapid prototyping and  visualization.


## Mathematical Foundation

The model is trained by optimizing the unweighted variational lower bound (ELBO):
```math
L_{\text{unweighted}}(\theta) = \mathbb{E}_{x_0,\epsilon,t} \left\| \epsilon - \epsilon_{\theta}(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \right\|_2^2
```

---

## Technical Deep Dive

I've written a comprehensive breakdown of the mathematics and architectural trade-offs of this project.
**Read the full blog post here:** [messi10tom.github.io/diffwave-diffusion](https://www.google.com/search?q=https://messi10tom.github.io/)
