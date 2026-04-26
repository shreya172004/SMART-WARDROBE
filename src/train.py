# ================================================================
# TRAINING SCRIPT — ViBE MODEL
# ================================================================
#
# BUGS FIXED vs previous version:
#
# Bug 1: OutfitBatchSampler applied to BodyClothDataset
#   OutfitBatchSampler does `_, outfit_id, _ = dataset[idx]`
#   but BodyClothDataset returns a dict {"body":..., "image":...}
#   → outfit_id received a tensor, not an int → broken grouping
#   Fix: OutfitBatchSampler removed from Stage 3 entirely.
#   BodyClothDataset has no outfit labels — random shuffle is correct.
#
# Bug 2: EarlyStopping direction inverted
#   stopper.step(-recall5, ...) with lower=better saved wrong epochs.
#   Fix: EarlyStopping(higher_is_better=True), pass recall5 directly.
#
# Bug 3: check_embedding_collapse not imported from loss.py
#   Redefined locally as check_embedding_health — diverged from loss.py.
#   Fix: import check_embedding_collapse from loss.py.
#
# Bug 4: WARNING threshold 0.12 too strict
#   Corrected to 0.2 in loss.py. Previous runs were actually healthy
#   by epoch 11 (rand_sim=0.08) but showed WARNING the whole time.
#
# ================================================================

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from dataset import BodyClothDataset
from model import ViBEModel
from loss import DeboasedInfoNCELoss, check_embedding_collapse


class EarlyStopping:
    def __init__(self, patience=5, save_path="best_model.pth",
                 higher_is_better=True):
        self.patience         = patience
        self.save_path        = save_path
        self.higher_is_better = higher_is_better
        self.best_value       = float("-inf") if higher_is_better else float("inf")
        self.counter          = 0
        self.best_epoch       = 0

    def step(self, value, model, epoch):
        improved = (value > self.best_value if self.higher_is_better
                    else value < self.best_value)
        if improved:
            self.best_value = value
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            direction = "up" if self.higher_is_better else "down"
            print(f"  Saved best model -> {self.save_path}  ({value:.4f} {direction})")
            return False
        self.counter += 1
        print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            print("  Early stopping triggered.")
            return True
        return False


@torch.no_grad()
def quick_recall_at5(model, loader, device, max_batches=20):
    model.eval()
    correct, total = 0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        body  = batch["body"].to(device, non_blocking=True)
        cloth = batch["image"].to(device, non_blocking=True)
        body_emb  = F.normalize(model.encode_body(body),  dim=1)
        cloth_emb = F.normalize(model.encode_cloth(cloth), dim=1)
        scores = torch.matmul(body_emb, cloth_emb.T)
        k      = min(5, scores.size(1))
        topk   = scores.topk(k=k, dim=1).indices
        labels = torch.arange(scores.size(0), device=device).unsqueeze(1)
        hits   = (topk == labels).any(dim=1)
        correct += hits.sum().item()
        total   += body.size(0)
    return correct / max(total, 1)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = BodyClothDataset(
        image_dir=config.TRAIN_DIR, body_csv=config.BODY_VECTOR_CSV, augment=True
    )
    val_dataset = BodyClothDataset(
        image_dir=config.VAL_DIR, body_csv=config.BODY_VECTOR_CSV, augment=False
    )
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples  : {len(val_dataset)}")

    # Standard DataLoader — OutfitBatchSampler removed (BodyClothDataset has no outfit IDs)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                               shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)

    model = ViBEModel().to(device)

    if os.path.exists(config.POLYVORE_ENCODER_PATH):
        print("  Loading Polyvore-pretrained clothing encoder")
        model.cloth_encoder.load_state_dict(
            torch.load(config.POLYVORE_ENCODER_PATH, map_location=device), strict=False)
    elif os.path.exists(config.DEEPFASHION_ENCODER_PATH):
        print("  Loading DeepFashion-pretrained clothing encoder")
        model.cloth_encoder.load_state_dict(
            torch.load(config.DEEPFASHION_ENCODER_PATH, map_location=device), strict=False)

    model.cloth_encoder.freeze_backbone()
    print("  Phase 1: backbone frozen")

    loss_fn = DeboasedInfoNCELoss(
        init_temperature=config.TEMPERATURE, fn_threshold=0.90, hard_neg_weight=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR, weight_decay=3e-4   #from 1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    stopper   = EarlyStopping(patience=config.EARLY_STOP_PATIENCE,
                               save_path=config.BEST_MODEL_PATH, higher_is_better=True)
    scaler    = GradScaler("cuda")

    train_losses, val_losses, recall5_scores = [], [], []

    for epoch in range(config.EPOCHS):

        if epoch == 3:
            model.cloth_encoder.unfreeze_layer4_only()
            print(f"\n  Phase 2: layer4 unfrozen (epoch {epoch+1})\n")

        model.train()
        train_loss = 0.0
        bar = tqdm(train_loader, desc=f"Train Ep {epoch+1}/{config.EPOCHS}")
        for batch in bar:
            body_vec  = batch["body"].to(device, non_blocking=True)
            cloth_img = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda"):
                body_emb, cloth_emb = model(body_vec, cloth_img)
                loss, temp = loss_fn(body_emb, cloth_emb, body_vec)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}", temp=f"{float(temp):.3f}")

        avg_train = train_loss / max(len(train_loader), 1)
        train_losses.append(avg_train)

        model.eval()
        val_loss, all_body_embs, all_cloth_embs = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                body_vec  = batch["body"].to(device, non_blocking=True)
                cloth_img = batch["image"].to(device, non_blocking=True)
                body_emb, cloth_emb = model(body_vec, cloth_img)
                loss, _ = loss_fn(body_emb, cloth_emb, body_vec)
                val_loss += loss.item()
                all_body_embs.append(body_emb.cpu())
                all_cloth_embs.append(cloth_emb.cpu())

        avg_val = val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val)

        print(f"\nEpoch {epoch+1}/{config.EPOCHS} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"Temp: {loss_fn.temperature.item():.4f}")

        check_embedding_collapse(
            torch.cat(all_body_embs, dim=0),
            torch.cat(all_cloth_embs, dim=0)
        )

        recall5 = quick_recall_at5(model, val_loader, device)
        recall5_scores.append(recall5)
        print(f"  Quick Recall@5: {recall5:.4f}")

        scheduler.step()
        if stopper.step(recall5, model, epoch):
            break

    n = len(train_losses)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(range(1,n+1), train_losses, label="Train")
    axes[0].plot(range(1,n+1), val_losses,   label="Val")
    axes[0].axvline(stopper.best_epoch+1, color="green", linestyle="--",
                    label=f"Best ep {stopper.best_epoch+1}")
    axes[0].set_title("Loss curves"); axes[0].legend()
    axes[1].plot(range(1,n+1), train_losses, label="Train")
    axes[1].plot(range(1,n+1), val_losses,   label="Val")
    axes[1].set_yscale("log"); axes[1].set_title("Loss (log)"); axes[1].legend()
    r = len(recall5_scores)
    axes[2].plot(range(1,r+1), recall5_scores, color="orange", marker="o", label="Recall@5")
    axes[2].set_title("Recall@5"); axes[2].legend()
    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/SmartWardrobe/loss_curves.png", dpi=120, bbox_inches="tight")
    print(f"\nTraining complete. Best epoch: {stopper.best_epoch+1}  Best Recall@5: {stopper.best_value:.4f}")

if __name__ == "__main__":
    train()