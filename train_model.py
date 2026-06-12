# # train_transformer.py
# """
# End-to-end fine-tuning script for the Student Transformer that uses a pretrained
# latent-flow bridge. This script:
#  - loads teacher (to copy early/late layers)
#  - loads tokenizer
#  - loads pretrained flow checkpoint (velocity model)
#  - assembles StudentTransformer (early -> flow-unroll -> late)
#  - fine-tunes (or trains) end-to-end for classification on ModelNet40

# Usage:
#     python train_transformer.py

# Notes:
#  - Paths to checkpoints and hyperparams live in CONFIG below; edit before running.
#  - This script provides options to freeze parts of the network during fine-tuning.
# """

# import os
# import time
# from pathlib import Path
# from pprint import pprint

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# # Import your project modules (adjust import paths if necessary)
# from data_loader import get_dataloader
# from point_transformer import PointTransformer
# from tokenizer import LatentTokenizer  # ensure your tokenizer class name matches
# from flow_model import TokenTransformerVelocity, sample_unroll
# from student_transformer import StudentTransformer, CrossAttentionDetokenizer

# # ----------------------------
# # CONFIG - edit paths/hyperparams
# # ----------------------------
# CONFIG = {
#     "data_root": "./data/modelnet40_h5",
#     "batch_size": 32,
#     "num_workers": 4,
#     "npoints": 1024,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",

#     "teacher": {
#         "in_channels": 3,
#         "embed_dim": 128,
#         "depth": 8,
#         "num_heads": 4,
#         "m_layer": 2,
#         "n_layer": 5
#     },

#     "tokenizer": {
#         "type": "cross_attn",   # "cross_attn" or "pointnet"
#         "dim": 128,
#         "num_tokens": 8,
#         "num_heads": 4
#     },

#     "flow": {
#         "checkpoint": "./checkpoints/flow/flow_epoch_20.pth",  # pretrained flow ckpt (optional)
#         "token_dim": 128,
#         "num_heads": 4,
#         "num_layers": 2,
#         "mlp_ratio": 2.0
#     },

#     "training": {
#         "epochs": 80,
#         "lr": 3e-4,
#         "weight_decay": 1e-2,
#         "out_dir": "./checkpoints/student",
#         "log_interval": 20,
#         "eval_interval": 1,
#         "save_every": 5,
#         "unroll_steps": 12,
#         "freeze_flow": False,      # keep True to freeze flow during fine-tune, False to allow updating
#         "freeze_early": False      # if True, freeze early layers
#     }
# }
# # ----------------------------


# def build_teacher(cfg, device):
#     t = cfg["teacher"]
#     teacher = PointTransformer(in_channels=t["in_channels"],
#                                embed_dim=t["embed_dim"],
#                                depth=t["depth"],
#                                num_heads=t["num_heads"]).to(device)
#     teacher.m_layer = t["m_layer"]
#     teacher.n_layer = t["n_layer"]
#     return teacher


# def build_tokenizer(cfg, device):
#     tc = cfg["tokenizer"]
#     # Here we assume LatentTokenizer class exists; PointNet encoder variant also acceptable
#     tokenizer = LatentTokenizer(input_dim=3, latent_dim=tc["dim"], output_points=cfg["npoints"]) \
#         if tc["type"] == "pointnet" else LatentTokenizer(dim=tc["dim"], num_tokens=tc["num_tokens"], num_heads=tc["num_heads"])
#     return tokenizer.to(device)


# def build_velocity(cfg, device):
#     f = cfg["flow"]
#     model = TokenTransformerVelocity(token_dim=f["token_dim"],
#                                      num_heads=f["num_heads"],
#                                      num_layers=f["num_layers"],
#                                      mlp_ratio=f["mlp_ratio"]).to(device)
#     # load pretrained flow if checkpoint provided
#     ckpt_path = f.get("checkpoint", None)
#     if ckpt_path and os.path.exists(ckpt_path):
#         ckpt = torch.load(ckpt_path, map_location=device)
#         state = ckpt.get("velocity_state", ckpt)
#         try:
#             model.load_state_dict(state)
#             print(f"[Flow] Loaded weights from {ckpt_path}")
#         except Exception as e:
#             print(f"[Flow] Failed to load complete state_dict: {e}; trying partial load")
#             model.load_state_dict(state, strict=False)
#     else:
#         print("[Flow] No pretrained checkpoint found or path invalid; training from scratch.")
#     return model


# def get_dataloaders(cfg):
#     train_loader = get_dataloader(root=cfg["data_root"], batch_size=cfg["batch_size"],
#                                   split='train', npoints=cfg["npoints"],
#                                   augment=True, num_workers=cfg["num_workers"])
#     val_loader = get_dataloader(root=cfg["data_root"], batch_size=cfg["batch_size"],
#                                 split='test', npoints=cfg["npoints"],
#                                 augment=False, num_workers=cfg["num_workers"])
#     return train_loader, val_loader


# def accuracy(preds, labels):
#     pred_labels = preds.argmax(dim=1)
#     return (pred_labels == labels).float().mean().item()


# def save_checkpoint(state, out_dir, epoch):
#     Path(out_dir).mkdir(parents=True, exist_ok=True)
#     path = os.path.join(out_dir, f"student_epoch_{epoch}.pth")
#     torch.save(state, path)
#     print(f"Saved checkpoint: {path}")


# def evaluate(model, dataloader, device):
#     model.eval()
#     total_acc = 0.0
#     total = 0
#     with torch.no_grad():
#         for points, labels in dataloader:
#             points = points.to(device)
#             labels = labels.to(device)
#             logits = model(points)
#             acc = accuracy(logits, labels)
#             bs = points.size(0)
#             total_acc += acc * bs
#             total += bs
#     model.train()
#     return total_acc / total if total > 0 else 0.0


# def main(cfg):
#     device = torch.device(cfg["device"])
#     pprint(cfg)
#     out_dir = cfg["training"]["out_dir"]
#     Path(out_dir).mkdir(parents=True, exist_ok=True)

#     # dataloaders
#     train_loader, val_loader = get_dataloaders(cfg)

#     # teacher and tokenizer and pretrained flow
#     teacher = build_teacher(cfg, device)
#     tokenizer = build_tokenizer(cfg, device)
#     velocity = build_velocity(cfg, device)

#     # IMPORTANT: teacher is only used to copy layers into student; we do not use teacher in forward pass
#     # If you want to load a pretrained teacher checkpoint, do it here (optional)
#     # teacher.load_state_dict(torch.load("path_to_teacher.pth"))

#     # build detokenizer manually with known dims: we assume token_dim == cfg["flow"]["token_dim"], point_dim == cfg["teacher"]["embed_dim"]
#     detok = CrossAttentionDetokenizer(token_dim=cfg["flow"]["token_dim"], point_dim=cfg["teacher"]["embed_dim"], num_heads=cfg["tokenizer"]["num_heads"]).to(device)

#     # build student model
#     student = StudentTransformer(teacher=teacher,
#                                  tokenizer=tokenizer,
#                                  velocity_model=velocity,
#                                  m_layer=cfg["teacher"]["m_layer"],
#                                  n_layer=cfg["teacher"]["n_layer"],
#                                  detokenizer=detok,
#                                  unroll_steps=cfg["training"]["unroll_steps"],
#                                  freeze_teacher_early=cfg["training"]["freeze_early"],
#                                  device=device).to(device)

#     print("Student model created. Params:")
#     total_params = sum(p.numel() for p in student.parameters())
#     trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
#     print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

#     # Optionally freeze flow weights
#     if cfg["training"]["freeze_flow"]:
#         for p in student.velocity_model.parameters():
#             p.requires_grad = False
#         print("[Train] Flow bridge frozen during fine-tuning.")

#     # optimizer: train entire student or select parts
#     optimizer = optim.AdamW([p for p in student.parameters() if p.requires_grad],
#                              lr=cfg["training"]["lr"],
#                              weight_decay=cfg["training"]["weight_decay"])

#     criterion = nn.CrossEntropyLoss()

#     # training loop
#     epochs = cfg["training"]["epochs"]
#     log_interval = cfg["training"]["log_interval"]
#     save_every = cfg["training"]["save_every"]
#     eval_interval = cfg["training"]["eval_interval"]

#     global_step = 0
#     for epoch in range(1, epochs + 1):
#         student.train()
#         epoch_loss = 0.0
#         epoch_acc = 0.0
#         t0 = time.time()
#         for batch_idx, (points, labels) in enumerate(train_loader):
#             points = points.to(device)
#             labels = labels.to(device)

#             logits = student(points)  # forward (includes tokenizer + unrolled flow)
#             loss = criterion(logits, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             batch_acc = accuracy(logits, labels)
#             epoch_loss += loss.item()
#             epoch_acc += batch_acc
#             global_step += 1

#             if global_step % log_interval == 0:
#                 print(f"[Epoch {epoch} Step {global_step}] batch {batch_idx} loss={loss.item():.4f} batch_acc={batch_acc:.4f}")

#         t1 = time.time()
#         avg_loss = epoch_loss / len(train_loader)
#         avg_acc = epoch_acc / len(train_loader)
#         print(f"Epoch {epoch} DONE. avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f} time={(t1-t0):.1f}s")

#         # evaluation
#         if epoch % eval_interval == 0:
#             val_acc = evaluate(student, val_loader, device)
#             print(f"[Eval] Epoch {epoch}: val_acc={val_acc:.4f}")

#         # save checkpoint
#         if epoch % save_every == 0:
#             ckpt = {
#                 "epoch": epoch,
#                 "student_state": student.state_dict(),
#                 "optimizer_state": optimizer.state_dict(),
#                 "cfg": cfg
#             }
#             save_checkpoint(ckpt, out_dir, epoch)

#     print("Fine-tuning finished.")


# if __name__ == "__main__":
#     main(CONFIG)


# train_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from point_transformer import PointTransformer
from tokenizer import Tokenizer
from flow_model import TokenTransformerVelocity
from student_transformer import StudentTransformer, CrossAttentionDetokenizer

# EDIT PATHS
DATA_ROOT = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\dataset\h5"
CHECKPOINT_DIR = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\checkpoints"
FLOW_DIR = os.path.join(CHECKPOINT_DIR, "flow")
os.makedirs(os.path.join(CHECKPOINT_DIR, "student"), exist_ok=True)

BATCH_SIZE = 8
NPOINTS = 1024
EPOCHS = 30
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Teacher checkpoint
TEACHER_CKPT = os.path.join(CHECKPOINT_DIR, "teacher_latest.pth")
FLOW_CKPT = os.path.join(FLOW_DIR, "flow_latest.pth")  # optional

if __name__ == "__main__":
    # data loaders
    train_loader = get_dataloader(root=DATA_ROOT, batch_size=BATCH_SIZE, split='train', npoints=NPOINTS, augment=True, num_workers=0)
    test_loader = get_dataloader(root=DATA_ROOT, batch_size=BATCH_SIZE, split='test', npoints=NPOINTS, augment=False, num_workers=0)

    # teacher
    teacher = PointTransformer(in_channels=3, embed_dim=128, depth=8, num_heads=4, num_classes=40).to(DEVICE)
    if os.path.exists(TEACHER_CKPT):
        ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE)
        teacher.load_state_dict(ckpt["model_state"])
        print("[INFO] Loaded teacher ckpt")
    teacher.eval()

    # build tokenizer & velocity, try load pretrained flow if exists
    tokenizer = Tokenizer(embed_dim=128, num_tokens=8).to(DEVICE)
    velocity = TokenTransformerVelocity(token_dim=128, num_heads=4, num_layers=2).to(DEVICE)
    if os.path.exists(FLOW_CKPT):
        flow_ckpt = torch.load(FLOW_CKPT, map_location=DEVICE)
        velocity.load_state_dict(flow_ckpt["velocity_state"])
        tokenizer.load_state_dict(flow_ckpt["tokenizer_state"])
        print("[INFO] Loaded pretrained flow and tokenizer")

    detok = CrossAttentionDetokenizer(token_dim=128, point_dim=128, num_heads=4).to(DEVICE)

    # assemble student
    student = StudentTransformer(teacher=teacher, tokenizer=tokenizer, velocity_model=velocity, m_layer=3, n_layer=5, detokenizer=detok, unroll_steps=12, freeze_early=False, freeze_flow=False, device=DEVICE).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR, weight_decay=1e-2)

    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for pts, labels in train_loader:
            pts, labels = pts.to(DEVICE), labels.to(DEVICE)
            logits = student(pts)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
        train_acc = 100.0 * total_correct / total_samples
        avg_loss = total_loss / len(train_loader)
        # validation
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for pts, labels in test_loader:
                pts, labels = pts.to(DEVICE), labels.to(DEVICE)
                logits = student(pts)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = 100.0 * correct / total
        print(f"[Epoch {epoch+1}] train_loss={avg_loss:.4f} train_acc={train_acc:.2f}% val_acc={val_acc:.2f}%")
        torch.save({"epoch": epoch, "student_state": student.state_dict(), "opt_state": optimizer.state_dict()}, os.path.join(CHECKPOINT_DIR, "student", f"student_epoch_{epoch+1}.pth"))
        torch.save({"epoch": epoch, "student_state": student.state_dict(), "opt_state": optimizer.state_dict()}, os.path.join(CHECKPOINT_DIR, "student", "student_latest.pth"))
    print("Student fine-tuning finished.")
