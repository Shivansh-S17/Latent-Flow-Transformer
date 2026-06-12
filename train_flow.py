# # train_flow.py
# """
# Train the latent flow velocity model using Flow Matching (Algorithm 1 from LFT).
# This script assumes you have:
#  - data_loader.py -> get_dataloader(...)
#  - point_transformer.py -> PointTransformer (teacher)
#  - tokenizer.py -> either LatentTokenizer (B,K,D) or PointNet-based encoder (B,D)
#  - flow_model.py -> TokenTransformerVelocity + flow_matching_loss + sample_unroll

# Usage:
#     python train_flow.py
# """

# import os
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from pathlib import Path
# from pprint import pprint

# # Import your modules (adjust imports if you placed files in subfolders)
# from data_loader import get_dataloader
# from point_transformer import PointTransformer
# from tokenizer import LatentTokenizer as CrossAttentionTokenizer  # if present
# from tokenizer import LatentTokenizer as PointNetTokenizer       # if the file contains PointNet variant
# from flow_model import TokenTransformerVelocity, flow_matching_loss, sample_unroll

# # ---------------------------
# # CONFIG (edit as needed)
# # ---------------------------
# CONFIG = {
#     "data_root": "./data/modelnet40_h5",   # folder containing ModelNet40 .h5 files
#     "batch_size": 32,
#     "num_workers": 4,
#     "npoints": 1024,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",

#     "teacher": {
#         "in_channels": 3,
#         "embed_dim": 128,
#         "depth": 8,
#         "num_heads": 4,
#         # which layers to take as x0 and x1 (0-indexed)
#         "m_layer": 2,
#         "n_layer": 5
#     },

#     "tokenizer": {
#         "type": "cross_attn",   # "cross_attn" or "pointnet"
#         "dim": 128,
#         "num_tokens": 8,        # K
#         "num_heads": 4
#     },

#     "flow": {
#         "token_dim": 128,       # D (must match tokenizer dim)
#         "num_heads": 4,
#         "num_layers": 2,
#         "mlp_ratio": 2.0
#     },

#     "optim": {
#         "lr": 1e-4,
#         "weight_decay": 1e-2
#     },

#     "training": {
#         "epochs": 20,
#         "save_every": 1,
#         "out_dir": "./checkpoints/flow",
#         "log_interval": 50
#     }
# }
# # ---------------------------


# def build_teacher(cfg):
#     tcfg = cfg["teacher"]
#     teacher = PointTransformer(in_channels=tcfg["in_channels"],
#                                embed_dim=tcfg["embed_dim"],
#                                depth=tcfg["depth"],
#                                num_heads=tcfg["num_heads"])
#     # store desired indices on the teacher for convenience
#     teacher.m_layer = tcfg["m_layer"]
#     teacher.n_layer = tcfg["n_layer"]
#     return teacher


# def build_tokenizer(cfg):
#     tcfg = cfg["tokenizer"]
#     if tcfg["type"] == "cross_attn":
#         # This is the cross-attention tokenizer we wrote earlier (returns B,K,D)
#         tokenizer = CrossAttentionTokenizer(dim=tcfg["dim"], num_tokens=tcfg["num_tokens"], num_heads=tcfg["num_heads"])
#     else:
#         # Use PointNet-style tokenizer (returns B, D)
#         tokenizer = PointNetTokenizer(input_dim=3, latent_dim=tcfg["dim"])
#     return tokenizer


# def prepare_dataloaders(cfg):
#     dl_train = get_dataloader(root=cfg["data_root"], batch_size=cfg["batch_size"], split='train',
#                               npoints=cfg["npoints"], augment=True, num_workers=cfg["num_workers"])
#     dl_val = get_dataloader(root=cfg["data_root"], batch_size=cfg["batch_size"], split='test',
#                             npoints=cfg["npoints"], augment=False, num_workers=cfg["num_workers"])
#     return dl_train, dl_val


# def ensure_dir(path):
#     Path(path).mkdir(parents=True, exist_ok=True)


# def main(cfg):
#     device = torch.device(cfg["device"])
#     print("Device:", device)
#     pprint(cfg)
#     ensure_dir(cfg["training"]["out_dir"])

#     # dataloaders
#     dl_train, dl_val = prepare_dataloaders(cfg)

#     # teacher model (frozen)
#     teacher = build_teacher(cfg).to(device)
#     # If you have a pretrained teacher checkpoint, load it here:
#     # teacher.load_state_dict(torch.load("path_to_teacher.pth"))
#     teacher.eval()
#     for p in teacher.parameters():
#         p.requires_grad = False

#     # tokenizer
#     tokenizer = build_tokenizer(cfg).to(device)
#     # tokenizer may be trainable; in the LFT paper the tokenization is often part of the student/flow pipeline.
#     # We'll keep tokenizer trainable (so it can adapt), but you can freeze it if you prefer.
#     tokenizer.train()

#     # velocity model
#     fcfg = cfg["flow"]
#     velocity_model = TokenTransformerVelocity(token_dim=fcfg["token_dim"],
#                                              num_heads=fcfg["num_heads"],
#                                              num_layers=fcfg["num_layers"],
#                                              mlp_ratio=fcfg["mlp_ratio"]).to(device)

#     optimizer = optim.AdamW(list(velocity_model.parameters()) + list(tokenizer.parameters()),
#                             lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

#     # Training loop
#     epochs = cfg["training"]["epochs"]
#     save_every = cfg["training"]["save_every"]
#     log_interval = cfg["training"]["log_interval"]
#     out_dir = cfg["training"]["out_dir"]

#     global_step = 0
#     for epoch in range(1, epochs + 1):
#         velocity_model.train()
#         tokenizer.train()
#         epoch_loss = 0.0
#         t0 = time.time()
#         for batch_idx, (points, labels) in enumerate(dl_train):
#             points = points.to(device)  # (B, N, 3)
#             B = points.shape[0]

#             # Run teacher to get hidden states at m and n
#             # The teacher.forward(points, return_layers=[m,n]) returns (logits, hidden_dict)
#             with torch.no_grad():
#                 _logits, hidden = teacher(points, return_layers=[teacher.m_layer, teacher.n_layer])
#                 # hidden entries: tensors of shape (B, N, D_pt)
#                 z0_pts = hidden[teacher.m_layer].detach()
#                 z1_pts = hidden[teacher.n_layer].detach()

#             # Tokenize: tokenizer may accept (B, N, D) and return either (B, K, D) or (B, D)
#             z0_tok = tokenizer(z0_pts)  # could be (B,K,D) or (B,D)
#             z1_tok = tokenizer(z1_pts)

#             # Normalize token outputs to consistent shape (B, K, D)
#             # If tokenizer returns (B,D) -> expand to (B,1,D)
#             if z0_tok.ndim == 2:
#                 z0_tok = z0_tok.unsqueeze(1)
#             if z1_tok.ndim == 2:
#                 z1_tok = z1_tok.unsqueeze(1)

#             # Ensure shapes
#             assert z0_tok.dim() == 3 and z1_tok.dim() == 3, "tokenizer must return (B,K,D) or (B,D)."
#             assert z0_tok.shape == z1_tok.shape, f"z0 and z1 tokens shapes mismatch: {z0_tok.shape} vs {z1_tok.shape}"

#             # Move tokens to device (they already are if teacher on same device)
#             z0_tok = z0_tok.to(device)
#             z1_tok = z1_tok.to(device)

#             # Compute flow matching loss
#             loss, info = flow_matching_loss(velocity_model, z0_tok, z1_tok, device=device)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
#             global_step += 1

#             if global_step % log_interval == 0:
#                 print(f"[Epoch {epoch}] Step {global_step} Batch {batch_idx}  loss={loss.item():.6f}  v_t_norm={info['v_target_norm']:.4f} v_pred_norm={info['v_pred_norm']:.4f}")

#         t1 = time.time()
#         avg_loss = epoch_loss / len(dl_train)
#         print(f"Epoch {epoch} completed. avg_loss={avg_loss:.6f}  time={(t1-t0):.1f}s")

#         # validation sanity check: sample flow outputs for a few validation batches and compute L2
#         velocity_model.eval()
#         tokenizer.eval()
#         with torch.no_grad():
#             val_loss = 0.0
#             nval = 0
#             for vbatch_idx, (vpoints, vlabels) in enumerate(dl_val):
#                 vpoints = vpoints.to(device)
#                 _, vhidden = teacher(vpoints, return_layers=[teacher.m_layer, teacher.n_layer])
#                 vz0_pts = vhidden[teacher.m_layer]
#                 vz1_pts = vhidden[teacher.n_layer]
#                 vz0_tok = tokenizer(vz0_pts)
#                 vz1_tok = tokenizer(vz1_pts)
#                 if vz0_tok.ndim == 2:
#                     vz0_tok = vz0_tok.unsqueeze(1)
#                 if vz1_tok.ndim == 2:
#                     vz1_tok = vz1_tok.unsqueeze(1)
#                 l, info_val = flow_matching_loss(velocity_model, vz0_tok, vz1_tok, device=device)
#                 val_loss += l.item()
#                 nval += 1
#                 # limit validation compute to few batches
#                 if nval >= 10:
#                     break
#             if nval > 0:
#                 print(f"Validation (first {nval} batches) mean loss: {val_loss / nval:.6f}")

#         # save checkpoint
#         if epoch % save_every == 0:
#             ckpt = {
#                 "epoch": epoch,
#                 "velocity_state": velocity_model.state_dict(),
#                 "tokenizer_state": tokenizer.state_dict(),
#                 "optimizer_state": optimizer.state_dict(),
#                 "cfg": cfg
#             }
#             save_path = os.path.join(out_dir, f"flow_epoch_{epoch}.pth")
#             torch.save(ckpt, save_path)
#             print(f"Saved checkpoint: {save_path}")

#     print("Flow training finished.")


# if __name__ == "__main__":
#     main(CONFIG)

# train_flow.py
"""
# Train the latent flow network between two teacher transformer layers.

# Steps:
# 1. Load pretrained teacher model.
# 2. Extract intermediate latents (z_m, z_n) using the tokenizer.
# 3. Train velocity model (u_theta) using flow-matching loss.

# Output:
# - Checkpoints of trained velocity model.
# - Flow loss logs for analysis.
# """

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm

# from data_loader import get_dataloader
# from point_transformer import PointTransformer
# from tokenizer import Tokenizer
# from flow_model import TokenTransformerVelocity, flow_matching_loss

# # ---------------- CONFIG ----------------
# DATA_ROOT = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\dataset\h5"
# CHECKPOINT_DIR = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\checkpoints\flow"
# TEACHER_CKPT = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\checkpoints\latest.pth"

# BATCH_SIZE = 8
# NPOINTS = 1024
# EPOCHS = 30
# LR = 1e-4
# LAYER_M = 3    # first layer to extract
# LAYER_N = 5    # second layer to extract
# K_TOKENS = 8   # number of latent tokens

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     print(f"[INFO] Using device: {DEVICE}")
#     print("[INFO] Preparing data...")
#     loader = get_dataloader(root=DATA_ROOT, batch_size=BATCH_SIZE, split='train', npoints=NPOINTS, augment=False, num_workers=0)

#     # 1️⃣ Load teacher model
#     print("[INFO] Loading pretrained teacher model...")
#     teacher = PointTransformer(num_classes=40).to(DEVICE)
#     ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE)
#     teacher.load_state_dict(ckpt["model_state"])
#     teacher.eval()

#     # 2️⃣ Initialize tokenizer and flow model
#     tokenizer = Tokenizer(embed_dim=64, num_tokens=K_TOKENS).to(DEVICE)
#     velocity_model = TokenTransformerVelocity(token_dim=64, num_heads=4, num_layers=2).to(DEVICE)
#     optimizer = optim.AdamW(list(tokenizer.parameters()) + list(velocity_model.parameters()), lr=LR)

#     print(f"[INFO] Flow training between layers {LAYER_M} and {LAYER_N}")

#     # 3️⃣ Training loop
#     for epoch in range(EPOCHS):
#         teacher.eval()
#         tokenizer.train()
#         velocity_model.train()

#         epoch_loss = 0.0
#         pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

#         for pts, _ in pbar:
#             pts = pts.to(DEVICE)
#             optimizer.zero_grad()

#             # Forward through teacher to extract layer activations
#             with torch.no_grad():
#                 _, hidden = teacher(pts, return_layers=[LAYER_M, LAYER_N])
#                 h_m, h_n = hidden[LAYER_M], hidden[LAYER_N]  # (B,N,D)

#             # Encode into latent tokens
#             z0 = tokenizer.encode(h_m)  # (B,K,D)
#             z1 = tokenizer.encode(h_n)  # (B,K,D)

#             # Flow matching loss
#             loss, info = flow_matching_loss(velocity_model, z0, z1, device=DEVICE)
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
#             pbar.set_postfix(loss=loss.item(), vnorm=info["v_pred_norm_mean"])

#         avg_loss = epoch_loss / len(loader)
#         print(f"[EPOCH {epoch+1}] Avg Flow Loss: {avg_loss:.6f}")

#         # Save checkpoint
#         ckpt_path = os.path.join(CHECKPOINT_DIR, f"flow_epoch_{epoch+1}.pth")
#         torch.save({
#             "epoch": epoch,
#             "velocity_state": velocity_model.state_dict(),
#             "tokenizer_state": tokenizer.state_dict()
#         }, ckpt_path)

#     print(f"\n[INFO] Flow model training completed. Checkpoints saved to {CHECKPOINT_DIR}")

# train_flow.py
import os
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tokenizer import Tokenizer
from flow_model import TokenTransformerVelocity, flow_matching_loss

# EDIT PATHS
CHECKPOINT_DIR = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\checkpoints"
LATENT_DIR = os.path.join(CHECKPOINT_DIR, "latents")
FLOW_OUT = os.path.join(CHECKPOINT_DIR, "flow")
os.makedirs(FLOW_OUT, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
K_TOKENS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use whichever latent file was saved by extract_latents.py (train split)
LATENT_FILE = None
for f in os.listdir(LATENT_DIR):
    if f.startswith("latents_train"):
        LATENT_FILE = os.path.join(LATENT_DIR, f)
        break
if LATENT_FILE is None:
    raise FileNotFoundError("No latent files found; run extract_latents.py first")

def load_latents(path):
    data = torch.load(path, map_location='cpu')
    return data['z_m'], data['z_n'], data['labels']

if __name__ == "__main__":
    print(f"[INFO] Loading latents from {LATENT_FILE}")
    z0_all, z1_all, labels = load_latents(LATENT_FILE)  # z0_all: (N_total, N_points, D)
    dataset = TensorDataset(z0_all, z1_all)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Build tokenizer (operates on per-point features to produce tokens)
    # Tokenizer expects embed_dim to match the hidden dim of the teacher layers
    embed_dim = z0_all.shape[-1]
    tokenizer = Tokenizer(embed_dim=embed_dim, num_tokens=K_TOKENS).to(DEVICE)
    velocity = TokenTransformerVelocity(token_dim=embed_dim, num_heads=4, num_layers=2).to(DEVICE)
    optimizer = optim.AdamW(list(tokenizer.parameters()) + list(velocity.parameters()), lr=LR, weight_decay=1e-2)

    for epoch in range(EPOCHS):
        tokenizer.train(); velocity.train()
        total_loss = 0.0
        for b, (z0_batch, z1_batch) in enumerate(loader):
            z0_batch = z0_batch.to(DEVICE)
            z1_batch = z1_batch.to(DEVICE)
            # encode per-point features -> tokens
            z0_tok = tokenizer.encode(z0_batch)  # (B, K, D)
            z1_tok = tokenizer.encode(z1_batch)
            loss, info = flow_matching_loss(velocity, z0_tok, z1_tok, device=DEVICE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"[Epoch {epoch+1}/{EPOCHS}] flow_loss={avg:.6f}")
        torch.save({"epoch": epoch, "velocity_state": velocity.state_dict(), "tokenizer_state": tokenizer.state_dict()}, os.path.join(FLOW_OUT, f"flow_epoch_{epoch+1}.pth"))
        torch.save({"epoch": epoch, "velocity_state": velocity.state_dict(), "tokenizer_state": tokenizer.state_dict()}, os.path.join(FLOW_OUT, "flow_latest.pth"))
    print("Flow training finished.")
