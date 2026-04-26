"""
train.py — Stage 3: Fashionista body-clothing alignment

Runs AFTER pretrain_clothing_encoder.py stages 1 and 2.
Loads Polyvore-pretrained clothing encoder and aligns it to body measurements.

Fixes in this version:
  1. Removes broken duplicate quick_recall_at5 definitions
  2. Uses dict-style batch access consistently: batch["body"], batch["image"]
  3. Adds working EarlyStopping.step()
  4. Adds robust quick Recall@5
  5. Adds embedding collapse diagnostics locally
  6. Keeps phase-1 freeze and phase-2 partial unfreeze
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from dataset import BodyClothDataset
from model import ViBEModel
from loss import DeboasedInfoNCELoss


# ================================================================
# HELPERS
# ================================================================
class EarlyStopping:
    """Stops training when monitored metric stops improving."""

    def __init__(self, patience: int = 3, save_path: str = "best_model.pth"):
        self.patience = patience
        self.save_path = save_path
        self.best_metric = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, metric_value, model, epoch):
        improved = metric_value < self.best_metric
        if improved:
            self.best_metric = metric_value
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            print(f"  Saved best model -> {self.save_path}")
            return False
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience


def check_embedding_collapse(body_embs: torch.Tensor, cloth_embs: torch.Tensor):
    """Simple embedding health check."""
    body_std = body_embs.std(dim=0).mean().item()
    cloth_std = cloth_embs.std(dim=0).mean().item()

    n = min(100, cloth_embs.size(0))
    if n >= 2:
        idx = torch.randperm(n)
        rand_cloth_sim = F.cosine_similarity(cloth_embs[:n], cloth_embs[idx], dim=1).mean().item()
    else:
        rand_cloth_sim = 1.0

    verdict = "OK"
    if body_std < 0.1 or cloth_std < 0.1 or rand_cloth_sim > 0.1:
        verdict = "WARNING"

    print("\n── Embedding health check ──────────────────")
    print(f"  Body std        : {body_std:.4f}  (want > 0.1)")
    print(f"  Cloth std       : {cloth_std:.4f}  (want > 0.1)")
    print(f"  Random cloth sim: {rand_cloth_sim:.4f}  (want < 0.1)")
    print(f"  Verdict         : {verdict}")
    print("────────────────────────────────────────────")


@torch.no_grad()
def quick_recall_at5(model, loader, device, max_batches=None):
    """
    Fast batch-local Recall@5 sanity metric.
    Assumes each body in a batch is paired with the cloth at the same batch index.
    This is a quick diagnostic, not a final gallery-wide retrieval metric.
    """
    model.eval()
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        body = batch["body"].to(device, non_blocking=True)
        cloth = batch["image"].to(device, non_blocking=True)

        body_emb = model.encode_body(body)
        cloth_emb = model.encode_cloth(cloth)

        body_emb = F.normalize(body_emb, dim=1)
        cloth_emb = F.normalize(cloth_emb, dim=1)

        scores = torch.matmul(body_emb, cloth_emb.T)
        k = min(5, scores.size(1))
        topk = scores.topk(k=k, dim=1).indices

        labels = torch.arange(scores.size(0), device=topk.device).unsqueeze(1)
        hits = (topk == labels).any(dim=1)

        correct += hits.sum().item()
        total += hits.numel()

    return correct / max(total, 1)


# ================================================================
# MAIN TRAINING FUNCTION
# ================================================================
def train(
    epochs: int = config.EPOCHS,
    save_path: str = config.BEST_MODEL_PATH,
    polyvore_ckpt: str = config.POLYVORE_ENCODER_PATH,
    deepfashion_ckpt: str = config.DEEPFASHION_ENCODER_PATH,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # ── Datasets ────────────────────────────────────────────────────
    train_dataset = BodyClothDataset(
        image_dir=config.TRAIN_DIR,
        body_csv=config.BODY_VECTOR_CSV,
        use_upper_crop=True,
        augment=True,
        body_completeness_threshold=0.7
    )
    val_dataset = BodyClothDataset(
        image_dir=config.VAL_DIR,
        body_csv=config.BODY_VECTOR_CSV,
        use_upper_crop=True,
        augment=False,
        body_completeness_threshold=0.7
    )

    print(f"  Dataset: {len(train_dataset)} valid samples from {config.TRAIN_DIR}")
    print(f"  Dataset: {len(val_dataset)} valid samples from {config.VAL_DIR}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ── Model ────────────────────────────────────────────────────────
    model = ViBEModel(
        body_input_dim=config.BODY_INPUT_DIM,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)

    if os.path.exists(polyvore_ckpt):
        print("  Loading Polyvore-pretrained clothing encoder")
        state = torch.load(polyvore_ckpt, map_location=device)
        model.cloth_encoder.load_state_dict(state, strict=False)
    elif os.path.exists(deepfashion_ckpt):
        print("  Loading DeepFashion-pretrained clothing encoder")
        state = torch.load(deepfashion_ckpt, map_location=device)
        model.cloth_encoder.load_state_dict(state, strict=False)
    else:
        print("  WARNING: No pretrained encoder found — using default initialization")

    # ── Phase 1 ─────────────────────────────────────────────────────
    model.cloth_encoder.freeze_backbone()
    print("  Phase 1: backbone frozen")

    # ── Loss ────────────────────────────────────────────────────────
    loss_fn = DeboasedInfoNCELoss(
        init_temperature=config.TEMPERATURE,
        fn_threshold=0.85,
        hard_neg_weight=0.3
    ).to(device)

    # ── Optimizer ───────────────────────────────────────────────────
    backbone_params = list(model.cloth_encoder.backbone.parameters())
    body_params = list(model.body_encoder.parameters())
    projector_params = list(model.cloth_encoder.projector.parameters())
    loss_params = list(loss_fn.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": body_params + projector_params + loss_params, "lr": config.LR},
            {"params": backbone_params, "lr": config.LR * 0.01},
        ],
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    stopper = EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        save_path=save_path
    )

    train_losses = []
    val_losses = []
    recall_history = []
    phase2_started = False

    # ── Training loop ───────────────────────────────────────────────
    for epoch in range(epochs):
        if epoch == 3 and not phase2_started:
            model.cloth_encoder.unfreeze_layer4_only()
            print(f"\n  Phase 2: layer4 unfrozen (epoch {epoch+1})\n")
            phase2_started = True

        # Train
        model.train()
        loss_fn.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Train Ep {epoch+1}/{epochs}")
        for batch in train_bar:
            body_vec = batch["body"].to(device, non_blocking=True)
            cloth_img = batch["image"].to(device, non_blocking=True)

            body_emb, cloth_emb = model(body_vec, cloth_img)
            loss, temp = loss_fn(body_emb, cloth_emb, body_vec)

            if not torch.isfinite(loss):
                print("  WARNING: non-finite train loss encountered, skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}", temp=f"{float(temp):.3f}")

        avg_train = train_loss / max(len(train_loader), 1)
        train_losses.append(avg_train)

        # Validate
        model.eval()
        loss_fn.eval()
        val_loss = 0.0
        all_body_embs = []
        all_cloth_embs = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Val   Ep {epoch+1}/{epochs}")
            for batch in val_bar:
                body_vec = batch["body"].to(device, non_blocking=True)
                cloth_img = batch["image"].to(device, non_blocking=True)

                body_emb, cloth_emb = model(body_vec, cloth_img)
                loss, _ = loss_fn(body_emb, cloth_emb, body_vec)

                if not torch.isfinite(loss):
                    print("  WARNING: non-finite val loss encountered, skipping batch")
                    continue

                val_loss += loss.item()
                all_body_embs.append(body_emb.cpu())
                all_cloth_embs.append(cloth_emb.cpu())

        avg_val = val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val)

        print(
            f"\nEpoch {epoch+1}/{epochs} | "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
            f"Temp: {loss_fn.temperature.item():.4f}"
        )

        if len(all_body_embs) > 0 and len(all_cloth_embs) > 0:
            check_embedding_collapse(
                torch.cat(all_body_embs, dim=0),
                torch.cat(all_cloth_embs, dim=0)
            )

        recall5 = quick_recall_at5(model, val_loader, device)
        recall_history.append(recall5)
        print(f"  Quick Recall@5: {recall5:.4f}")

        scheduler.step()

        # Early stop on Recall@5 improvement; negate so "lower is better" logic still works
        if stopper.step(-recall5, model, epoch):
            print("  Early stopping triggered.")
            break

    # ── Plots ────────────────────────────────────────────────────────
    out_dir = "/content/drive/MyDrive/SmartWardrobe"
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.axvline(stopper.best_epoch, color="green", linestyle="--",
                label=f"Best ep {stopper.best_epoch+1}")
    plt.title("Loss curves")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.yscale("log")
    plt.title("Loss curves (log)")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(recall_history, label="Recall@5")
    plt.axvline(stopper.best_epoch, color="green", linestyle="--",
                label=f"Best ep {stopper.best_epoch+1}")
    plt.title("Quick Recall@5")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(plot_path, dpi=120)
    plt.show()

    print(
        f"\n✓ Training complete. Best epoch: {stopper.best_epoch + 1}, "
        f"best monitored metric: {stopper.best_metric:.4f}"
    )
    print(f"  Plot saved -> {plot_path}")


if __name__ == "__main__":
    train()