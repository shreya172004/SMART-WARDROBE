"""
train.py — Stage 3: Fashionista body-clothing alignment (FULLY FIXED)

Key fixes:
  - Fixed all NameError/UnpackError crashes
  - Dataset dict access → tuple unpack
  - Proper model method names
  - Removed broken nested class
  - Added missing check_embedding_collapse
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
# MISSING: check_embedding_collapse (add this)
# ================================================================
def check_embedding_collapse(body_embs, cloth_embs):
    """Print embedding health diagnostics."""
    body_std = body_embs.std(dim=0).mean().item()
    cloth_std = cloth_embs.std(dim=0).mean().item()
    
    n = min(100, body_embs.size(0))
    idx = torch.randperm(n)
    rand_cloth_sim = F.cosine_similarity(
        cloth_embs[:n], cloth_embs[idx]
    ).mean().item()
    
    verdict = "OK" if rand_cloth_sim < 0.1 else "WARNING"
    print(f"\n── Embedding health check ──────────────────")
    print(f"  Body std        : {body_std:.4f}  (want > 0.1)")
    print(f"  Cloth std       : {cloth_std:.4f}  (want > 0.1)")
    print(f"  Random cloth sim: {rand_cloth_sim:.4f}  (want < 0.1)")
    print(f"  Verdict         : {verdict}")
    print("────────────────────────────────────────────")

# FIXED: EarlyStopping (no nested methods)
class EarlyStopping:
    """Stops training when val loss stops improving."""
    def __init__(self, patience: int = 3, save_path: str = "best_model.pth"):
        self.patience  = patience
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter   = 0
        self.best_epoch = 0

    def step(self, val_metric, model, epoch):
        if val_metric < self.best_loss:
            self.best_loss = val_metric
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            print(f"  Saved best model (epoch {epoch+1})")
            return False
        self.counter += 1
        return self.counter >= self.patience

# FIXED: quick_recall_at5 (proper unpacking, normalization, tensor ops)
def quick_recall_at5(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:  # ← FIXED: no unpacking here
            # FIXED: your dataloader returns (body, image)
            body_vec, cloth_img = batch[0].to(device), batch[1].to(device)
            
            body_emb = F.normalize(model.body_encoder(body_vec), dim=1)
            cloth_emb = F.normalize(model.cloth_encoder(cloth_img), dim=1)

            scores = torch.matmul(body_emb, cloth_emb.T)
            top5 = scores.topk(k=min(5, scores.size(1)), dim=1).indices

            for i in range(body_emb.size(0)):
                # FIXED: proper tensor comparison
                if (top5[i] == i).any().item():
                    correct += 1
                total += 1

    return correct / max(total, 1)

# ================================================================
# MAIN TRAINING FUNCTION (minor fixes)
# ================================================================
def train(
    epochs:           int   = config.EPOCHS,
    save_path:        str   = config.BEST_MODEL_PATH,
    polyvore_ckpt:    str   = config.POLYVORE_ENCODER_PATH,
    deepfashion_ckpt: str   = config.DEEPFASHION_ENCODER_PATH,
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

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Dataset: {len(train_dataset)} valid samples from {config.TRAIN_DIR}")
    print(f"  Dataset: {len(val_dataset)} valid samples from {config.VAL_DIR}")

    # ── Model ────────────────────────────────────────────────────────
    model = ViBEModel(
        body_input_dim=config.BODY_INPUT_DIM,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)

    # Load pretrained clothing encoder
    if os.path.exists(polyvore_ckpt):
        print(f"  Loading Polyvore-pretrained clothing encoder")
        model.cloth_encoder.load_state_dict(
            torch.load(polyvore_ckpt, map_location=device), strict=False)
    elif os.path.exists(deepfashion_ckpt):
        print(f"  Loading DeepFashion-pretrained clothing encoder")
        model.cloth_encoder.load_state_dict(
            torch.load(deepfashion_ckpt, map_location=device), strict=False)
    else:
        print("  WARNING: No pretrained encoder found — using ImageNet weights only")

    # ── Phase 1: freeze backbone ─────────────────────────────────────
    model.cloth_encoder.freeze_backbone()
    print(f"  Phase 1: backbone frozen")

    # ── Loss & optimizer ─────────────────────────────────────────────
    loss_fn = DeboasedInfoNCELoss(
        init_temperature=config.TEMPERATURE,
        fn_threshold=0.85,
        hard_neg_weight=0.3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    stopper = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, save_path=save_path)

    train_losses, val_losses = [], []
    phase2_started = False

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(epochs):

        # Phase 2: unfreeze layer4 at epoch 3
        if epoch == 3 and not phase2_started:
            model.cloth_encoder.unfreeze_layer4_only()
            print(f"\n  Phase 2: layer4 unfrozen (epoch {epoch+1})\n")
            phase2_started = True

        # ── Train ──────────────────────────────────────────────────
        model.train()
        loss_fn.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Train Ep {epoch+1}/{epochs}")
        for batch in loop:
            # FIXED: tuple unpacking instead of dict access
            body_vec, cloth_img = batch[0].to(device), batch[1].to(device)

            body_emb, cloth_emb = model(body_vec, cloth_img)
            loss, temp = loss_fn(body_emb, cloth_emb, body_vec)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", temp=f"{temp:.3f}")

        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        loss_fn.eval()
        val_loss = 0.0
        all_body_embs = []
        all_cloth_embs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val   Ep {epoch+1}/{epochs}"):
                # FIXED: tuple unpacking
                body_vec, cloth_img = batch[0].to(device), batch[1].to(device)

                body_emb, cloth_emb = model(body_vec, cloth_img)
                loss, _ = loss_fn(body_emb, cloth_emb, body_vec)
                val_loss += loss.item()

                all_body_embs.append(body_emb.cpu())
                all_cloth_embs.append(cloth_emb.cpu())

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"Temp: {loss_fn.temperature.item():.4f}")

        # Collapse check
        check_embedding_collapse(
            torch.cat(all_body_embs, dim=0),
            torch.cat(all_cloth_embs, dim=0)
        )

        scheduler.step()

        # FIXED: call works now
        recall5 = quick_recall_at5(model, val_loader, device)
        print(f"  Quick Recall@5: {recall5:.4f}")

        if stopper.step(-recall5, model, epoch):
            break

    # ── Plots ────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.axvline(stopper.best_epoch, color="green", linestyle="--",
                label=f"Best (ep {stopper.best_epoch+1})")
    plt.title("Loss curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.yscale("log")
    plt.title("Loss curves (log scale)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/SmartWardrobe/loss_curves.png", dpi=120)
    plt.show()

    print(f"\n✓ Training complete. Best epoch: {stopper.best_epoch+1}, "
          f"best val loss: {stopper.best_loss:.4f}")

if __name__ == "__main__":
    train()