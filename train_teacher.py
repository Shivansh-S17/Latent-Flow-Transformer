# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# from data_loader import get_dataloader
# from point_transformer import PointTransformer

# # ---------------- CONFIG ----------------
# DATA_ROOT = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\dataset\h5"
# CHECKPOINT_DIR = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\checkpoints"

# BATCH_SIZE = 16
# NPOINTS = 1024
# EPOCHS = 30
# LR = 1e-3
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Ensure checkpoint directory exists
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# # ---------------- TRAIN FUNCTION ----------------
# def train_one_epoch(model, loader, criterion, optimizer, epoch):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     loop = tqdm(loader, desc=f"Epoch [{epoch+1}] Training", leave=False)
#     for pts, labels in loop:
#         pts, labels = pts.to(DEVICE), labels.squeeze().to(DEVICE)
#         optimizer.zero_grad()

#         outputs = model(pts)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         preds = torch.argmax(outputs, dim=1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#         loop.set_postfix(loss=loss.item())

#     acc = 100.0 * correct / total
#     avg_loss = running_loss / len(loader)
#     return avg_loss, acc


# # ---------------- TEST FUNCTION ----------------
# def evaluate(model, loader, criterion):
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for pts, labels in loader:
#             pts, labels = pts.to(DEVICE), labels.squeeze().to(DEVICE)
#             outputs = model(pts)
#             loss = criterion(outputs, labels)

#             total_loss += loss.item()
#             preds = torch.argmax(outputs, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     acc = 100.0 * correct / total
#     avg_loss = total_loss / len(loader)
#     return avg_loss, acc


# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     print(f"[INFO] Using device: {DEVICE}")
#     print("[INFO] Loading data...")

#     train_loader = get_dataloader(root=DATA_ROOT, batch_size=BATCH_SIZE, split='train', npoints=NPOINTS, augment=True, num_workers=0)
#     test_loader = get_dataloader(root=DATA_ROOT, batch_size=BATCH_SIZE, split='test', npoints=NPOINTS, augment=False, num_workers=0)

#     print("[INFO] Initializing model...")
#     model = PointTransformer(num_classes=40).to(DEVICE)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR)

#     # Resume if checkpoint exists
#     start_epoch = 0
#     latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pth")
#     if os.path.exists(latest_ckpt):
#         print(f"[INFO] Loading checkpoint: {latest_ckpt}")
#         checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
#         model.load_state_dict(checkpoint["model_state"])
#         optimizer.load_state_dict(checkpoint["optimizer_state"])
#         start_epoch = checkpoint["epoch"] + 1

#     # ---------------- TRAIN LOOP ----------------
#     for epoch in range(start_epoch, EPOCHS):
#         train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
#         test_loss, test_acc = evaluate(model, test_loader, criterion)

#         print(f"\n[Epoch {epoch+1}/{EPOCHS}] "
#               f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
#               f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

#         # ---- Save checkpoint ----
#         checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
#         torch.save({
#             "epoch": epoch,
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict()
#         }, checkpoint_path)

#         # Save as latest
#         torch.save({
#             "epoch": epoch,
#             "model_state": model.state_dict(),
#             "optimizer_state": optimizer.state_dict()
#         }, latest_ckpt)

#     print(f"\n[INFO] Training completed! Checkpoints saved to: {CHECKPOINT_DIR}")

# train_teacher.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_loader import get_dataloader
from point_transformer import PointTransformer

# EDIT THESE paths to match your system
DATA_ROOT = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\dataset\h5"
CHECKPOINT_DIR = r"C:\Users\shiva\OneDrive - iitgn.ac.in\Desktop\LFT-Point\checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 8
NPOINTS = 1024
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc="Train", leave=False)
    for pts, labels in loop:
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(pts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item())
    return running_loss / len(loader), 100.0 * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for pts, labels in loader:
            pts, labels = pts.to(DEVICE), labels.to(DEVICE)
            logits = model(pts)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total

if __name__ == "__main__":
    train_loader = get_dataloader(root=DATA_ROOT, batch_size=BATCH_SIZE, split='train', npoints=NPOINTS, augment=True, num_workers=0)
    test_loader = get_dataloader(root=DATA_ROOT, batch_size=BATCH_SIZE, split='test', npoints=NPOINTS, augment=False, num_workers=0)

    model = PointTransformer(in_channels=3, embed_dim=128, depth=8, num_heads=4, num_classes=40).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    start_epoch = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "teacher_latest.pth")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[INFO] Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%")
        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"teacher_epoch_{epoch+1}.pth"))
        torch.save(ckpt, latest_ckpt)
    print("Teacher training finished.")
