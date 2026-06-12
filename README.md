# Latent Flow Transformer for Point Cloud Classification

## Layer Compression via Latent Flow Matching (LFM)

This repository implements a **Latent Flow Transformer (LFT)** framework for compressing deep point cloud transformer models using **Flow Matching in latent space**.

The proposed method replaces multiple transformer layers with a learned continuous latent flow, significantly reducing **network depth, parameters, and FLOPs** while preserving classification accuracy.

---

# Project Pipeline

The complete training pipeline consists of four stages:

1. **Train a Deep Transformer (Teacher Model)**
2. **Extract Hidden Representations from Layers `m` and `n`**
3. **Train a Latent Velocity Network using Flow Matching**
4. **Replace Intermediate Transformer Layers with the Learned Flow (Student Model)**

```
Teacher Transformer
        │
        ▼
Extract Hidden States (z_m, z_n)
        │
        ▼
Flow Matching Training
        │
        ▼
Learn Latent Velocity Network
        │
        ▼
Compressed Student Transformer
```

---

# Mathematical Formulation

Let

* (z_m) = hidden representation at layer **m**
* (z_n) = hidden representation at layer **n**

We define a linear interpolation between them:

[
z_t = (1-t)z_m + tz_n
]

where

[
t \sim U(0,1)
]

---

## Target Velocity

The target velocity is defined as

[
v_{\text{target}} = z_n - z_m
]

---

## Flow Matching Loss

The velocity network (v_\theta(z_t,t)) is trained using

[
\mathcal{L}_{FM}
================

\left|
v_\theta(z_t,t)
---------------

(z_n-z_m)
\right|^2
]

This teaches the network how to continuously transform (z_m) into (z_n) in latent space.

---

# Inference

During inference, integration starts from the hidden state (z_m).

For **K Euler integration steps**

[
z_{k+1}
=======

z_k
+
\frac{1}{K}
v_\theta(z_k,t_k)
]

### Single-Step Approximation

If

[
K=1
]

then

[
z_n
\approx
z_m
+
v_\theta(z_m,1)
]

Multi-step Euler integration generally provides a more accurate approximation.

---

# Repository Structure

```
├── data_loader.py
├── point_transformer.py
├── extract_latents.py
├── flow_model.py
├── tokenizer.py
├── train_teacher.py
├── train_flow.py
├── student_transformer.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Create Environment

```bash
conda create -n lft python=3.9
conda activate lft
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

* Python 3.9
* PyTorch
* NumPy
* h5py
* tqdm
* thop

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

# Dataset

The experiments use **ModelNet40** stored in H5 format.

```
dataset/
└── h5/
    ├── train_*.h5
    ├── test_*.h5
```

Update the `DATA_ROOT` variable inside

* `train_teacher.py`
* `extract_latents.py`
* `train_flow.py`
* `train_model.py`

before training.

---

# Training Pipeline

Run the following steps in order.

---

## Step 1 — Train Teacher Model

Train the original full-depth transformer.

```bash
python train_teacher.py
```

### Output

```
CheckPoint/
└── teacher_latest.pth
```

---

## Step 2 — Extract Latent Features

Extract hidden representations from layers **m** and **n**.

```bash
python extract_latents.py
```

### Output

```
CheckPoint/
└── latents/
    ├── latents_train_mX_nY.pt
    └── latents_test_mX_nY.pt
```

Each file contains

* `z_m`
* `z_n`
* `labels`

---

## Step 3 — Train Flow Model

Train the latent velocity network using Flow Matching.

```bash
python train_flow.py
```

### Output

```
CheckPoint/
└── flow/
    └── flow_latest.pth
```

---

## Step 4 — Train Compressed Student Model

Train the final compressed transformer.

```bash
python train_model.py
```

Architecture:

```
Input
   │
   ▼
Transformer Layers (0 → m)
   │
   ▼
Latent Flow Bridge
   │
   ▼
Transformer Layers (n → End)
   │
   ▼
Classification Head
```

### Output

```
CheckPoint/
└── student/
    └── student_best.pth
```

---

# Example Compression Settings

## Point Transformer (48 Layers)

| Parameter         | Value |
| ----------------- | ----- |
| m                 | 12    |
| n                 | 36    |
| Removed Layers    | 24    |
| Depth Compression | ~50%  |

---

## PCT (24 Layers)

| Parameter         | Value |
| ----------------- | ----- |
| m                 | 6     |
| n                 | 16    |
| Removed Layers    | 10    |
| Depth Compression | ~45%  |

---

# Expected Results

| Model       | Parameters | FLOPs    | Accuracy   |
| ----------- | ---------- | -------- | ---------- |
| Teacher     | High       | High     | Baseline   |
| LFT Student | ↓ 40–50%   | ↓ 50–60% | Comparable |

---

# Important Hyperparameters

Modify these values inside the training scripts:

```python
m_layer
n_layer
unroll_steps
learning_rate
warmup_epochs
max_grad_norm
```

**Important:** The values of `m_layer` and `n_layer` must remain consistent across

* `extract_latents.py`
* `train_flow.py`
* `train_model.py`

---

# Key Contributions

* ✅ Transformer layer compression via latent ODE/Flow modeling
* ✅ Flow Matching in transformer latent space
* ✅ Designed for 3D point cloud classification
* ✅ Reduced depth, parameters, and FLOPs
* ✅ Minimal performance degradation
* ✅ Plug-and-play compression framework for deep point transformers

---

# Citation

If you use this repository in your research, please consider citing:

```bibtex
@misc{latentflowtransformer2026,
  title={Latent Flow Transformer for Point Cloud Classification},
  author={Your Name},
  year={2026}
}
```

---

# License

This project is released under the MIT License.

---

# Acknowledgements

This work builds upon recent advances in:

* Point Transformer architectures
* Continuous Normalizing Flows
* Flow Matching
* Latent ODE models
* 3D Point Cloud Learning
