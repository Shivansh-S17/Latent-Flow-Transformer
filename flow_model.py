# # flow_model.py
# """
# Flow model and training helpers for Latent Flow Transformer (LFT) style training.

# Design notes (aligned with LFT paper):
# - Work in latent token space: inputs are z shaped (B, K, D).
# - Time conditioning: scalar t in [0,1] is embedded and added to token embeddings.
# - The network outputs an updated latent; velocity estimate is out - z (as in paper).
# - Flow matching loss uses straight-line interpolation:
#       z_t = (1-t) * z0 + t * z1
#       v_target = z1 - z0
#   and minimizes E_t || u_theta(z_t, t) - v_target ||^2

# This file contains:
# - TimeEmbedding: sinusoidal/MLP time embedding
# - TokenTransformerVelocity: token transformer velocity estimator
# - flow_matching_loss: computes the loss for a batch of (z0, z1)
# - sample_unroll: integrates the learned velocity from z0 -> zT via Euler steps
# """

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # -------------------------
# # Time embedding
# # -------------------------
# class TimeEmbedding(nn.Module):
#     """
#     Small time embedding MLP. Input t scalar (B,) or (B,1), output (B, D).
#     """
#     def __init__(self, dim, hidden_dim=None):
#         super().__init__()
#         if hidden_dim is None:
#             hidden_dim = dim * 2
#         self.net = nn.Sequential(
#             nn.Linear(1, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim)
#         )

#     def forward(self, t):
#         # t: (B,) or (B,1)
#         if t.ndim == 1:
#             t = t.unsqueeze(-1)
#         return self.net(t.float())


# # -------------------------
# # Small Transformer-based velocity estimator
# # -------------------------
# class TokenTransformerVelocity(nn.Module):
#     """
#     Velocity estimator u_theta for token latents.
#     Produces updated token embeddings 'out' and uses velocity = out - z as the estimate.
#     - z: (B, K, D)
#     - t: (B,) in [0,1]
#     Returns:
#     - v_pred: (B, K, D)  (predicted velocity)
#     """

#     def __init__(self, token_dim=128, num_heads=4, num_layers=2, mlp_ratio=2.0, time_dim=None, dropout=0.0):
#         super().__init__()
#         D = token_dim
#         self.token_dim = D
#         self.num_layers = num_layers
#         self.time_dim = time_dim if time_dim is not None else D
#         self.time_emb = TimeEmbedding(self.time_dim)

#         # A learnable linear projection to add time embedding per token
#         self.time_proj = nn.Linear(self.time_dim, D)

#         # small transformer encoder operating on K tokens
#         encoder_layer = nn.TransformerEncoderLayer(d_model=D, nhead=num_heads, dim_feedforward=int(D * mlp_ratio), activation='gelu', batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # final projection to token dimension
#         self.out_proj = nn.Linear(D, D)

#         # optionally layernorm
#         self.norm = nn.LayerNorm(D)

#     def forward(self, z, t):
#         """
#         z: (B, K, D)
#         t: (B,) scalar time values in [0,1]
#         returns:
#             v_pred: (B, K, D) predicted velocity field
#         """
#         B, K, D = z.shape
#         # time embedding: (B, D_time) -> project to D and add to each token
#         te = self.time_emb(t)                       # (B, time_dim)
#         te = self.time_proj(te)                     # (B, D)
#         te = te.unsqueeze(1).expand(-1, K, -1)      # (B, K, D)

#         # input to transformer: z + time embedding
#         h = z + te
#         # optional normalization
#         h = self.norm(h)

#         # transformer expects (B, seq_len, D) with batch_first=True
#         h = self.transformer(h)                     # (B, K, D)

#         out = self.out_proj(h)                      # (B, K, D)

#         # velocity estimate (paper: subtract input latent from augmented output)
#         v_pred = out - z                             # (B, K, D)
#         return v_pred


# # -------------------------
# # Flow matching loss
# # -------------------------
# def flow_matching_loss(velocity_model, z0, z1, device=None, exact_target=None):
#     """
#     Compute flow matching loss on a batch of pairs (z0, z1).
#     Implements the interpolation strategy from LFT: sample t ~ Uniform(0,1),
#     compute z_t = (1-t) z0 + t z1, and target velocity v = z1 - z0 (constant).
#     Minimize E || u_theta(z_t, t) - v ||^2.

#     Args:
#       velocity_model: instance of TokenTransformerVelocity (or similar); expects (z_t, t) -> v_pred
#       z0: Tensor (B, K, D)  - latent at layer m
#       z1: Tensor (B, K, D)  - latent at layer n (target)
#       device: (optional) torch device
#       exact_target: if provided, use this as v_target to override z1 - z0 (rare)
#     Returns:
#       loss: scalar tensor
#       info: dict with some diagnostics (mean norms)
#     """
#     if device is None:
#         device = z0.device
#     B = z0.shape[0]

#     # sample t uniformly per example
#     t = torch.rand(B, device=device)  # (B,)
#     # construct z_t along straight line
#     t_broadcast = t.view(B, 1, 1)      # (B,1,1)
#     zt = (1.0 - t_broadcast) * z0 + t_broadcast * z1  # (B, K, D)

#     v_target = z1 - z0 if exact_target is None else exact_target  # (B, K, D)

#     v_pred = velocity_model(zt, t)  # (B, K, D)

#     loss = F.mse_loss(v_pred, v_target)
#     info = {
#         'loss': loss.item(),
#         'v_target_norm': v_target.norm(dim=-1).mean().item(),
#         'v_pred_norm': v_pred.norm(dim=-1).mean().item()
#     }
#     return loss, info


# # -------------------------
# # Sampling / Unrolling
# # -------------------------
# def sample_unroll(velocity_model, z0, steps=20, device=None):
#     """
#     Integrate learned velocity from z0 (at t=0) to t=1 using simple Euler unrolling.
#     z_{t+dt} = z_t + dt * u_theta(z_t, t)

#     Args:
#       velocity_model: u_theta
#       z0: initial latent (B, K, D)
#       steps: number of Euler steps to take between t=0..1
#       device: torch device
#     Returns:
#       zT: (B, K, D) final latent after unroll
#     """
#     if device is None:
#         device = z0.device
#     dt = 1.0 / float(steps)
#     z = z0.clone()
#     B = z.shape[0]
#     for i in range(steps):
#         t = torch.full((B,), float(i) / steps, device=device)  # sample mid or left point
#         v = velocity_model(z, t)   # (B, K, D)
#         z = z + dt * v
#     return z


# # -------------------------
# # Training helper (example one-step)
# # -------------------------
# def step_flow_training(velocity_model, optimizer, teacher_encoder, tokenizer, data_batch, device):
#     """
#     One training step for the flow model following the paper's Algorithm 1:
#     - teacher_encoder: full transformer (frozen) that returns hidden states at layers m and n
#     - tokenizer: maps per-point features -> token latents (B, K, D)
#     - data_batch: tuple (points, labels) from dataloader
#     """
#     velocity_model.train()
#     optimizer.zero_grad()

#     points, _ = data_batch
#     points = points.to(device)

#     # teacher run: get hidden states at layers m and n
#     # teacher_encoder should be prepared to return hidden dict {m : tensor, n : tensor}
#     _, hidden = teacher_encoder(points, return_layers=[teacher_encoder.m_layer, teacher_encoder.n_layer])
#     # hidden[m] and hidden[n] shapes: (B, N, D_pt)
#     z0_pts = hidden[teacher_encoder.m_layer]
#     z1_pts = hidden[teacher_encoder.n_layer]

#     # tokenize (map to K tokens) -> (B, K, D)
#     z0 = tokenizer(z0_pts)
#     z1 = tokenizer(z1_pts)

#     # compute loss and step
#     loss, info = flow_matching_loss(velocity_model, z0, z1, device=device)
#     loss.backward()
#     optimizer.step()

#     return loss.item(), info


# # -------------------------
# # Quick sanity check
# # -------------------------
# if __name__ == "__main__":
#     B, K, D = 4, 8, 128
#     z0 = torch.randn(B, K, D)
#     z1 = torch.randn(B, K, D)
#     model = TokenTransformerVelocity(token_dim=D, num_heads=4, num_layers=2)
#     loss, info = flow_matching_loss(model, z0, z1)
#     print("Loss:", loss.item(), "info:", info)
#     zT = sample_unroll(model, z0, steps=10)
#     print("zT shape:", zT.shape)

# train_flow.py
"""
Train the latent flow network between two teacher transformer layers.

Steps:
1. Load pretrained teacher model.
2. Extract intermediate latents (z_m, z_n) using the tokenizer.
3. Train velocity model (u_theta) using flow-matching loss.

Output:
- Checkpoints of trained velocity model.
- Flow loss logs for analysis.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import get_dataloader
from point_transformer import PointTransformer
from tokenizer import Tokenizer
from flow_model import TokenTransformerVelocity, flow_matching_loss

# ---------------- CONFIG ----------------
DATA_ROOT = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\dataset\h5"
CHECKPOINT_DIR = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\checkpoints\flow"
TEACHER_CKPT = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\checkpoints\latest.pth"

BATCH_SIZE = 8
NPOINTS = 1024
EPOCHS = 30
LR = 1e-4
LAYER_M = 3    # first layer to extract
LAYER_N = 5    # second layer to extract
K_TOKENS = 8   # number of latent tokens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print(f"[INFO] Using device: {DEVICE}")
    print("[INFO] Preparing data...")
    loader = get_dataloader(root=DATA_ROOT, batch_size=BATCH_SIZE, split='train', npoints=NPOINTS, augment=False, num_workers=0)

    # 1️⃣ Load teacher model
    print("[INFO] Loading pretrained teacher model...")
    teacher = PointTransformer(num_classes=40).to(DEVICE)
    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()

    # 2️⃣ Initialize tokenizer and flow model
    tokenizer = Tokenizer(embed_dim=64, num_tokens=K_TOKENS).to(DEVICE)
    velocity_model = TokenTransformerVelocity(token_dim=64, num_heads=4, num_layers=2).to(DEVICE)
    optimizer = optim.AdamW(list(tokenizer.parameters()) + list(velocity_model.parameters()), lr=LR)

    print(f"[INFO] Flow training between layers {LAYER_M} and {LAYER_N}")

    # 3️⃣ Training loop
    for epoch in range(EPOCHS):
        teacher.eval()
        tokenizer.train()
        velocity_model.train()

        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for pts, _ in pbar:
            pts = pts.to(DEVICE)
            optimizer.zero_grad()

            # Forward through teacher to extract layer activations
            with torch.no_grad():
                _, hidden = teacher(pts, return_layers=[LAYER_M, LAYER_N])
                h_m, h_n = hidden[LAYER_M], hidden[LAYER_N]  # (B,N,D)

            # Encode into latent tokens
            z0 = tokenizer.encode(h_m)  # (B,K,D)
            z1 = tokenizer.encode(h_n)  # (B,K,D)

            # Flow matching loss
            loss, info = flow_matching_loss(velocity_model, z0, z1, device=DEVICE)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), vnorm=info["v_pred_norm_mean"])

        avg_loss = epoch_loss / len(loader)
        print(f"[EPOCH {epoch+1}] Avg Flow Loss: {avg_loss:.6f}")

        # Save checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"flow_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "velocity_state": velocity_model.state_dict(),
            "tokenizer_state": tokenizer.state_dict()
        }, ckpt_path)

    print(f"\n[INFO] Flow model training completed. Checkpoints saved to {CHECKPOINT_DIR}")
