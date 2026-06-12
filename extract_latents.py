# extract_latents.py
import os
import torch
from data_loader import get_dataloader
from point_transformer import PointTransformer

# EDIT PATHS
DATA_ROOT = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\dataset\h5"
CHECKPOINT_DIR = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\checkpoints"
OUT_DIR = os.path.join(CHECKPOINT_DIR, "latents")
os.makedirs(OUT_DIR, exist_ok=True)
TEACHER_CKPT = os.path.join(CHECKPOINT_DIR, "teacher_latest.pth")

BATCH_SIZE = 8
NPOINTS = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# choose the same layer indices you plan to compress
M_LAYER = 3
N_LAYER = 5

def extract(split='train'):
    model = PointTransformer(in_channels=3, embed_dim=128, depth=8, num_heads=4, num_classes=40).to(DEVICE)
    ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = get_dataloader(root=DATA_ROOT, batch_size=BATCH_SIZE, split=split, npoints=NPOINTS, augment=False, num_workers=0)

    zs_m = []
    zs_n = []
    labels_all = []
    with torch.no_grad():
        for pts, labels in loader:
            pts = pts.to(DEVICE)
            _, hidden = model(pts, return_layers=[M_LAYER, N_LAYER])
            z_m = hidden[M_LAYER].cpu()  # (B, N, D)
            z_n = hidden[N_LAYER].cpu()
            zs_m.append(z_m)
            zs_n.append(z_n)
            labels_all.append(labels)
    z_m_all = torch.cat(zs_m, dim=0)
    z_n_all = torch.cat(zs_n, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    out_path = os.path.join(OUT_DIR, f"latents_{split}_m{M_LAYER}_n{N_LAYER}.pt")
    torch.save({"z_m": z_m_all, "z_n": z_n_all, "labels": labels_all}, out_path)
    print(f"[INFO] Saved {split} latents -> {out_path}")

if __name__ == "__main__":
    extract('train')
    extract('test')
