# Improved Mamba SSM — AI for IoT Final Project

This project reproduces the Mamba Selective State Space Model and introduces several lightweight architectural improvements.  
Work completed by **John Orina**.

---

## Project Goals
- Reproduce a working Mamba implementation  
- Understand the internal design of SSMs  
- Propose and implement meaningful improvements  
- Compare baseline vs modified behavior

---

## Baseline
We used a minimal PyTorch implementation (`johnma2006/mamba-minimal`), which replicates the official Mamba architecture.
We validated the baseline using pretrained weights from HuggingFace.

---

## Improvements Implemented
### 1. **Residual Gating**
A learnable gate controlling skip-path contribution.

### 2. **Output Layer Normalization**
Stabilizes activations before projection.

### 3. **Dropout**
Improves training behavior and regularization.

### 4. **GELU Activations**
More expressive nonlinearities than SiLU.

These changes materially alter the computational graph.

---

## Results
- Baseline model produces coherent text.  
- Modified model produces noisy output when loaded with original weights  
  → **expected and correct**, since architecture changed.  
- Verified that forward pass runs successfully.

See `baseline_outputs/` and `improved_outputs/` for examples.

---

## 🧪 Running the Demo

### 1. Install dependencies
```bash
pip install torch einops transformers jupyter
