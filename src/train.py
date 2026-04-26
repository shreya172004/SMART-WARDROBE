# ================================================================
# TRAINING SCRIPT — ViBE MODEL (FINAL VERSION)
# ================================================================

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import config
from dataset import BodyClothDataset
from model import ViBEModel
from loss import DeboasedInfoNCELoss


# ================================================================
# EARLY STOPPING
# ================================================================
class EarlyStopping:
    def __init__(self, patience=5, save_path="best_model.pth"):
        self.patience = patience
        self.save_path = save_path
        self.best_metric = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, metric_value, model, epoch):
        if metric_value < self.best_metric:
            self.best_metric = metric_value
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            print(f"  Saved best model -> {self.save_path}")
            return False

        self.counter += 1
        print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience


# ================================================================
# QUICK RECALL@5 (FAST VERSION)
# ================================================================
@torch.no_grad()
def quick_recall_at5(model, loader, device, max_batches=20):
    model.eval()
    correct, total = 0, 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        body = batch["body"].to(device, non_blocking=True)
        cloth = batch["image"].to(device, non_blocking=True)

        # Normalize embeddings explicitly (IMPORTANT)
        body_emb = F.normalize(model.encode_body(body), dim=1)
        cloth_emb = F.normalize(model.encode_cloth(cloth), dim=1)

        scores = torch.matmul(body_emb, cloth_emb.T)
        k = min(5, scores.size(1))
        topk = scores.topk(k=k, dim=1).indices

        labels = torch.arange(scores.size(0), device=device).unsqueeze(1)
        hits = (topk == labels).any(dim=1)

        correct += hits.sum().item()
        total += hits.numel()

    return correct / max(total, 1)


# ================================================================
# EMBEDDING HEALTH CHECK
# ================================================================
def check_embedding_health(body_embs, cloth_embs):
    body_std = body_embs.std(dim=0).mean().item()
    cloth_std = cloth_embs.std(dim=0).mean().item()

    n = min(100, cloth_embs.size(0))
    perm = torch.randperm(n)
    rand_sim = F.cosine_similarity(cloth_embs[:n], cloth_embs[perm], dim=1).mean().item()

    verdict = "OK" if (body_std > 0.1 and cloth_std > 0.1 and rand_sim < 0.12) else "WARNING"

    print("\n── Embedding health check ──────────────────")
    print(f"  Body std        : {body_std:.4f}  (want > 0.1)")
    print(f"  Cloth std       : {cloth_std:.4f}  (want > 0.1)")
    print(f"  Random cloth sim: {rand_sim:.4f}  (want < 0.12)")
    print(f"  Verdict         : {verdict}")
    print("────────────────────────────────────────────")


# ================================================================
# MAIN TRAIN FUNCTION
# ================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # DATASETS
    # -------------------------
    train_dataset = BodyClothDataset(
        image_dir=config.TRAIN_DIR,
        body_csv=config.BODY_VECTOR_CSV,
        augment=True
    )

    val_dataset = BodyClothDataset(
        image_dir=config.VAL_DIR,
        body_csv=config.BODY_VECTOR_CSV,
        augment=False
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples  : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------
    # MODEL
    # -------------------------
    model = ViBEModel().to(device)

    # Load pretrained encoder
    if os.path.exists(config.POLYVORE_ENCODER_PATH):
        print("  Loading Polyvore-pretrained clothing encoder")
        state = torch.load(config.POLYVORE_ENCODER_PATH, map_location=device)
        model.cloth_encoder.load_state_dict(state, strict=False)

    elif os.path.exists(config.DEEPFASHION_ENCODER_PATH):
        print("  Loading DeepFashion-pretrained clothing encoder")
        state = torch.load(config.DEEPFASHION_ENCODER_PATH, map_location=device)
        model.cloth_encoder.load_state_dict(state, strict=False)

    model.cloth_encoder.freeze_backbone()
    print("  Phase 1: backbone frozen")

    # -------------------------
    # LOSS + OPTIMIZER
    # -------------------------
    loss_fn = DeboasedInfoNCELoss(
        init_temperature=config.TEMPERATURE,
        fn_threshold=0.90,
        hard_neg_weight=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    stopper = EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        save_path=config.BEST_MODEL_PATH
    )

    scaler = GradScaler()

    # ============================================================
    # TRAIN LOOP
    # ============================================================
    for epoch in range(config.EPOCHS):

        # Unfreeze layer4 after warmup
        if epoch == 3:
            model.cloth_encoder.unfreeze_layer4_only()
            print(f"\n  Phase 2: layer4 unfrozen (epoch {epoch+1})\n")

        # -------------------------
        # TRAIN
        # -------------------------
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Train Ep {epoch+1}/{config.EPOCHS}")

        for batch in train_bar:
            body_vec = batch["body"].to(device, non_blocking=True)
            cloth_img = batch["image"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                body_emb, cloth_emb = model(body_vec, cloth_img)
                loss, temp = loss_fn(body_emb, cloth_emb, body_vec)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}", temp=f"{float(temp):.3f}")

        avg_train = train_loss / max(len(train_loader), 1)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        val_loss = 0.0
        all_body_embs, all_cloth_embs = [], []

        with torch.no_grad():
            for batch in val_loader:
                body_vec = batch["body"].to(device, non_blocking=True)
                cloth_img = batch["image"].to(device, non_blocking=True)

                body_emb, cloth_emb = model(body_vec, cloth_img)
                loss, _ = loss_fn(body_emb, cloth_emb, body_vec)

                val_loss += loss.item()
                all_body_embs.append(body_emb.cpu())
                all_cloth_embs.append(cloth_emb.cpu())

        avg_val = val_loss / max(len(val_loader), 1)

        print(
            f"\nEpoch {epoch+1}/{config.EPOCHS} | "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
            f"Temp: {loss_fn.temperature.item():.4f}"
        )

        # -------------------------
        # HEALTH CHECK
        # -------------------------
        body_embs = torch.cat(all_body_embs, dim=0)
        cloth_embs = torch.cat(all_cloth_embs, dim=0)
        check_embedding_health(body_embs, cloth_embs)

        # -------------------------
        # RECALL
        # -------------------------
        recall5 = quick_recall_at5(model, val_loader, device)
        print(f"  Quick Recall@5: {recall5:.4f}")

        scheduler.step()

        # -------------------------
        # EARLY STOPPING
        # -------------------------
        if stopper.step(-recall5, model, epoch):
            print("  Early stopping triggered.")
            break

    print(f"\n✓ Training complete. Best epoch: {stopper.best_epoch + 1}")


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    train()