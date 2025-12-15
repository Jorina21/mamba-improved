# Improved Mamba SSM — AI for IoT Final Project

This project implements and studies a minimal version of the Mamba Selective State Space Model (SSM) in PyTorch.
The goal is to understand the Mamba architecture, explore lightweight architectural modifications, and evaluate
language modeling behavior under realistic compute constraints.

Work completed by **John Orina**.

---

## Project Goals
- Implement a working Mamba-style SSM from scratch
- Understand why SSMs scale linearly with sequence length
- Experiment with architectural stabilizations
- Study the effect of dataset scale and tokenization on output quality
- Train and evaluate the model on real language data

---

## Baseline Model
We began with a minimal PyTorch implementation inspired by open-source Mamba reference code.
This implementation preserves the core selective state space mechanics while remaining readable
and trainable on a single GPU.

Pretrained HuggingFace weights were initially used for validation, but the final experiments
focus on training from scratch.

---

## Architectural Improvements
The following lightweight modifications were added to improve training stability:

1. **Residual Gating**
   - Learnable gate controlling residual contribution.

2. **Output Layer Normalization**
   - Stabilizes activations before projection.

3. **Dropout**
   - Improves regularization during training.

4. **GELU Activations**
   - Replaces SiLU for improved expressiveness.

These changes intentionally modify the computational graph, making pretrained weights incompatible.

---

## Dataset and Tokenization
To study scaling behavior, the model was trained from scratch on the **TinyStories** dataset.
A **byte-level tokenizer (256 tokens)** was used to avoid vocabulary engineering and enable
general-purpose text modeling.

This setup allows the model to learn character, word, and sentence structure directly from data.

---

## Training Setup
- Model size: ~34M parameters
- Dataset: TinyStories (≈50–100M tokens)
- Tokenization: byte-level
- Hardware: Purdue Scholar GPU cluster
- Training performed in multiple resumed jobs

---

## Results
- Early training shows rapid loss reduction, confirming correct model execution.
- Loss stabilizes as the model converges for a given scale.
- Generated text transitions from random bytes to readable English with increasing training steps.
- Compared to official Mamba models, output quality is limited by scale, not correctness.

---

## Limitations
- This implementation does not use fused CUDA kernels.
- Training is slower than the official Mamba implementation.
- Model scale is several orders of magnitude smaller than production models.

These limitations are intentional and allow controlled experimentation on modest hardware.

---

## Running Training
```bash
python train.py --dataset data/TinyStoriesV2-GPT4-train.txt
```
---

## Demo and Presentation
A recorded walkthrough of the project, including model architecture, training setup,
and generated text examples, is available at the link below:

🔗 https://purdue0-my.sharepoint.com/:v:/g/personal/aleistne_purdue_edu/IQAnGuzL40uQQYGpBzbM5ydVAdf_9gmGZRQArW7JLsh-g88
