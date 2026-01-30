# ☕ Swin-Coffee: SOTA-Level Efficiency with Feature Denoising

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org)
[![Dataset: CIFAR-100](https://img.shields.io/badge/Dataset-CIFAR--100-blue.svg)]()
[![Efficiency: SOTA](https://img.shields.io/badge/Efficiency-30s%2Fepoch-green)]()

**Swin-transformer-coffee** (Swin-Coffee) is a highly optimized Vision Transformer architecture tailored for small-scale datasets like **CIFAR-100**. Addressing the challenge that Transformers struggle to converge from scratch on small data, we introduce **Convolutional Inductive Bias** and **Self-Supervised Feature Denoising**.

Achieving **82.36% Top-1 Accuracy** on CIFAR-100 **from scratch** (no pre-training), Swin-Coffee outperforms traditional CNN baselines while maintaining SOTA-level training efficiency (**30s/epoch** on 4x V100).

---

## 🏆 Core Competitiveness: "David vs. Goliath"

Swin-Coffee redefines the **Accuracy-Efficiency Trade-off**. By upsampling inputs to **64x64** and using a stride-2 Stem, we process features at a high-quality 32x32 resolution with extremely low computational cost.

### 1. Benchmark: Swin-Coffee vs. SOTA CNNs (CIFAR-100)
> **Setting:** All models trained **from scratch** (No ImageNet Pre-training).

| Model | Params | Training Speed (4xV100) | Top-1 Acc | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **ResNet-50** | 25.6M | ~45s / epoch | ~79.0% | Lacks receptive field for small objects. |
| **WideResNet-28-10** | 36.5M | ~60s / epoch | 81.50% | **Heavier.** 10x more FLOPs than ours. |
| **PyramidNet-272** | 26.0M | ~150s / epoch | **83.40%** | **Too Slow.** 272 layers make it impractical for edge deployment. |
| **Swin-Coffee (Ours)** | **26.8M** | **30s / epoch** 🚀 | **82.36%** | **The Efficiency King.** Beats WRN accuracy with 2x speed. |

### 2. Internal Improvement: +16.48% Gain
Compared to the standard Swin-Tiny baseline, our architectural innovations yield massive gains without external data.

| Model | Resolution | Epochs | Method | Top-1 Acc |
| :--- | :---: | :---: | :--- | :---: |
| Swin-Tiny (Baseline) | 224x224 | 300 | Standard | 65.88% |
| **Swin-Coffee** | **64x64** | **300-400** | **Scratch + Denoise** | **82.36%** (+16.48%) |

---

## 💡 Key Architectural Innovations

Swin-Coffee is not just a stack of layers; it's a "Denoising & Robustness" pipeline.

![Architecture Diagram](swin-coffee.jpg)

### 🚀 1. Strategic Resolution Processing (The "Sweet Spot")
* **Input**: Upsampled to **64x64** to preserve small object details.
* **Stem**: Uses a `Conv3x3 (stride=2)` to immediately reduce features to **32x32**.
* **Benefit**: We gain the information density of high-res inputs but keep the computational cost (FLOPs) extremely low—**approx. 1/10th of WideResNet-28-10**.

### 🧠 2. Swin-CBAM Fusion (Stage 1-3)
* **Problem**: Pure Transformers lack "inductive bias" (don't understand local edges well).
* **Solution**: We integrate **CBAM (Convolutional Block Attention Module)** after Swin blocks.
    * **Spatial Attention**: Uses convolution to enforce local connectivity.
    * **Channel Attention**: Dynamically recalibrates feature importance.

### 🌊 3. Enhanced Disrupt Block (FFT-based)
* **Mechanism**: Applied at the end of early stages. It performs **Frequency Domain Masking** using FFT.
* **Effect**: Randomly masks high/low-frequency components, forcing the model to learn robust structural features rather than memorizing textures (prevents overfitting).

### 🧹 4. Self-Supervised Denoising (Stage 4)
* **Mechanism**: **Late-Phase Denoising**.
    * After Epoch 30, we inject Gaussian noise into the high-level features.
    * The model must predict the correct class **AND** reconstruct the clean features (MSE Loss).
* **Effect**: Acts as a strong regularizer, ensuring the final embedding is noise-invariant and highly discriminative.

---

## 🛠️ Project Structure

Current release corresponds to the **82.36%** SOTA checkpoint.

```text
Swin-Coffee/
├── swin_coffee.py        # Core Model (SwinCBAM, Disrupt, Denoise Blocks)
├── swin-coffee.jpg       # Architecture Diagram
├── logs/                 # Training Logs
│   ├── training_log_swin_coffee  # Log for 81.92% -> 82.36% run
│   └── training_log_swin_tiny    # Baseline log
└── weights/              # Pre-trained weights (Coming Soon)
