"""
train.py — Stage 3: Fashionista body-clothing alignment
 
Runs AFTER pretrain_clothing_encoder.py stages 1 and 2.
Loads Polyvore-pretrained clothing encoder and aligns it to body measurements.
 
Key fixes vs v1:
  1. Temperature fixed to 0.07 (was 0.03 → caused loss collapse)
  2. DeboasedInfoNCELoss replaces final_loss() — mask-based debiasing,
     no gradient scaling instability
  3. unfreeze_layer4_only() replaces full unfreeze — stops overfitting
  4. Early stopping with patience=3 — stops at the real best epoch
  5. Collapse check after every val epoch
  6. Upper-body crop in dataset — fixes the heels recommendation issue
  7. Cosine LR scheduler replaces flat LR
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
from loss import DeboasedInfoNCELoss, check_embedding_collapse
 
 
# ================================================================
# HELPERS
# ================================================================
 
class EarlyStopping:
    """Stops training when val loss stops improving."""
 
    def __init__(self, patience: int = 3, save_path: str = "best_model.pth"):
        self.patience   = patience
        self.save_path  = save_path
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0
 
    def quick_recall_at5(model, val_loader, device, n_batches=10):
        """Fast Recall@5 estimate on first n_batches of val set."""
        model.eval()
        body_embs, cloth_embs = [], []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= n_batches: break
                b = model.encode_body(batch["body"].to(device)).cpu()
                c = model.encode_cloth(batch["image"].to(device)).cpu()
                body_embs.append(b)
                cloth_embs.append(c)

        body_embs  = torch.cat(body_embs)   # (N, 128)
        cloth_embs = torch.cat(cloth_embs)  # (N, 128)
        sims       = torch.matmul(body_embs, cloth_embs.T)  # (N, N)
        labels     = torch.arange(len(body_embs))
        top5       = sims.topk(5, dim=1).indices             # (N, 5)
        hits       = (top5 == labels.unsqueeze(1)).any(dim=1)
        return hits.float().mean().item()
 
 
# ================================================================
# MAIN TRAINING FUNCTION
# ================================================================
 
def train(
    epochs:              int   = config.EPOCHS,
    save_path:           str   = config.BEST_MODEL_PATH,
    polyvore_ckpt:       str   = config.POLYVORE_ENCODER_PATH,
    deepfashion_ckpt:    str   = config.DEEPFASHION_ENCODER_PATH,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
 
    # ── Datasets ────────────────────────────────────────────────────
    train_dataset = BodyClothDataset(
        image_dir=config.TRAIN_DIR,
        body_csv=config.BODY_VECTOR_CSV,
        use_upper_crop=True,    # KEY FIX: reduces pose/shoe bias
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
 
    train_loader = DataLoader(train_dataset,
                               batch_size=config.BATCH_SIZE,
                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,
                               batch_size=config.BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)
 
    # ── Model ────────────────────────────────────────────────────────
    model = ViBEModel(
        body_input_dim=config.BODY_INPUT_DIM,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)
 
    # Load the best available pretrained clothing encoder:
    # Polyvore (compatibility-aware) > DeepFashion (visual only) > ImageNet
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
 
    # ── Phase 1: freeze backbone, train body encoder + projector ─────
    model.cloth_encoder.freeze_backbone()
    print(f"  Phase 1: backbone frozen")
 
    # ── Loss & optimizer ─────────────────────────────────────────────
    loss_fn = DeboasedInfoNCELoss(
        init_temperature=config.TEMPERATURE,   # 0.07
        fn_threshold=0.85,
        hard_neg_weight=0.3
    ).to(device)
 
    # Separate LRs: body encoder + loss params get higher LR
    backbone_params   = list(model.cloth_encoder.backbone.parameters())
    non_backbone_params = (
        [p for p in model.body_encoder.parameters()]
        + list(model.cloth_encoder.projector.parameters())
        + list(loss_fn.parameters())
    )
 
    optimizer = torch.optim.AdamW([
        {"params": non_backbone_params, "lr": config.LR},
        {"params": backbone_params,     "lr": config.LR * 0.01},  # tiny initially
    ], weight_decay=1e-4)
 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
 
    stopper = EarlyStopping(patience=config.EARLY_STOP_PATIENCE,
                            save_path=save_path)
 
    train_losses, val_losses = [], []
    phase2_started = False
 
    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(epochs):
 
        # Phase 2: unfreeze layer4 at epoch 3
        if epoch == 3 and not phase2_started:
            model.cloth_encoder.unfreeze_layer4_only()
            # Boost backbone LR to allow layer4 to adapt
            for g in optimizer.param_groups:
                if g["params"][0] in backbone_params:
                    g["lr"] = config.LR * 0.1
            print(f"\n  Phase 2: layer4 unfrozen (epoch {epoch+1})\n")
            phase2_started = True
 
        # ── Train ──────────────────────────────────────────────────
        model.train()
        loss_fn.train()
        train_loss = 0.0
 
        loop = tqdm(train_loader, desc=f"Train Ep {epoch+1}/{epochs}")
        for batch in loop:
            body_vec  = batch["body"].to(device)
            cloth_img = batch["image"].to(device)
 
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
        all_body_embs  = []
        all_cloth_embs = []
 
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val   Ep {epoch+1}/{epochs}"):
                body_vec  = batch["body"].to(device)
                cloth_img = batch["image"].to(device)
 
                body_emb, cloth_emb = model(body_vec, cloth_img)
                loss, _  = loss_fn(body_emb, cloth_emb, body_vec)
                val_loss += loss.item()
 
                all_body_embs.append(body_emb.cpu())
                all_cloth_embs.append(cloth_emb.cpu())
 
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
 
        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"Temp: {loss_fn.temperature.item():.4f}")
 
        # Collapse check every epoch
        check_embedding_collapse(
            torch.cat(all_body_embs,  dim=0),
            torch.cat(all_cloth_embs, dim=0)
        )
 
        scheduler.step()
 
        recall5 = quick_recall_at5(model, val_loader, device)
        print(f"  Quick Recall@5: {recall5:.4f}")
        if stopper.step(-recall5, model, epoch): break 
 
    # ── Plots ────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.axvline(stopper.best_epoch, color="green", linestyle="--",
                label=f"Best (ep {stopper.best_epoch+1})")
    plt.title("Loss curves")
    plt.legend()
 
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
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
 

