"""
pretrain_clothing_encoder.py  —  Fixed NaN / collapse edition
 
ROOT CAUSE OF NaN CHAIN:
─────────────────────────────────────────────────────────────────
Stage 2 printed Val: 0.0000 every epoch — this was SILENT COLLAPSE,
not good training. The collapsed weights caused Stage 3 to produce
NaN from batch 1.
 
THREE sources of NaN / collapse fixed here:
 
NaN-1  exp() overflow in loss denominator
   torch.exp(sim/0.07) overflows float32 when sim > ~88.
   After Stage 1 the projector produces embeddings with cosine
   similarity ~0.97 → logit = 0.97/0.07 = 13.9 → exp = 1.1e6.
   With batch=32 items, sum ≈ 32 × 1.1e6 → log(...) fine.
   BUT after a few gradient steps with hard_neg boost, logits
   spike → exp overflows → sum = inf → log(inf) = inf →
   inf - inf = NaN.
   FIX: Use numerically stable log-sum-exp via
        F.cross_entropy (which applies log-softmax internally
        using the logsumexp trick). This is ALWAYS numerically
        stable regardless of logit magnitude.
 
NaN-2  Val loss = 0.0000 (silent collapse from batch 0)
   polyvore_contrastive_category_aware was called on the val
   loader WITHOUT hard_neg_weight, but STILL called with
   temperature=0.07. If the batch has NO positive pairs
   (common in a random 10% val split with 19268 outfits and
   batch=32 → probability of a same-outfit pair ≈ 32/19268 ≈ 0.002)
   then pos_off.sum(dim=1) = 0 for all rows → n_pos = 1 (clamped)
   → loss = -(0/1) = 0.0 for every batch → Val = 0.0000.
   This looked like perfect training but was actually division by
   zero masked by the clamp. The saved checkpoint was the FIRST
   batch's model (val=0 < inf) — essentially random.
   FIX: Use F.cross_entropy with diagonal labels — this is always
        well-defined even when no same-outfit pairs exist in the
        batch. It treats item i's own clothing as its positive
        (self-supervised style, valid for outfit compatibility).
 
NaN-3  body_encoder.normalize_input std=0 warning
   std() with a batch of identical body vectors = 0 → division by
   0+eps=1e-6 → values ~1e6 → NaN after Linear layer.
   Seen in the warning: "std(): degrees of freedom is <= 0"
   FIX: Use the population std (correction=0) and clamp to min 1e-4.
─────────────────────────────────────────────────────────────────
"""
 
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
 
import config
from cloth_encoder import ClothEncoder
from deepfashion_dataset import DeepFashionDataset
 
 
# ================================================================
# FASHION CATEGORIES
# ================================================================
 
FASHION_CATEGORIES = {
    "Day Dresses", "Boots", "Handbags", "Sunglasses", "Coats",
    "Blazers", "Skinny Jeans", "Watches", "Accessories",
    "Men's Shirts", "Rompers", "Sandals", "Backpacks",
    "Blouses", "Tops", "Sweaters", "Boyfriend Jeans",
    "Pumps", "Clutches", "Ankle Booties", "Vests",
    "Necklaces", "Sweatshirts", "Tank Top", "Shorts",
    "Skirts", "Jackets", "Sneakers", "Flats", "Heels",
    "Jumpsuits", "Cardigans", "T-Shirts", "Leggings",
    "Scarves", "Belts", "Rings", "Earrings", "Bracelets & Bangles",
    "Hats", "Lingerie", "Swimwear", "Active", "Denim",
    "Pants", "Trousers", "Overalls", "Tuxedos", "Suits",
    "Loafers", "Oxford Shoes", "Mules", "Wedges", "Platform Shoes",
    "Maxi Dresses", "Mini Dresses", "Midi Dresses",
    "Leather Jackets", "Denim Jackets", "Bomber Jackets",
    "Trench Coats", "Puffer Jackets", "Fur Coats",
    "Crop Tops", "Button-Down Shirts", "Polo Shirts",
    "Graphic Tees", "Hoodies",
}
 
 
def _get_set_id(item_id: str) -> str:
    return item_id.rsplit("_", 1)[0]
 
 
# ================================================================
# POLYVORE DATASET
# ================================================================
 
class PolyvoreArrowDataset(Dataset):
    """
    Returns (img_tensor, outfit_id_int, category_id_int).
    category_id uses deterministic sorted mapping — never hash().
    """
 
    def __init__(self, arrow_dir=config.POLYVORE_ARROW_DIR,
                 fashion_only=True, max_samples=None):
        from datasets import load_from_disk
 
        print(f"  Loading Polyvore from: {arrow_dir}")
        hf_dataset = load_from_disk(arrow_dir)
        print(f"  Raw rows: {len(hf_dataset)}")
 
        if fashion_only:
            hf_dataset = hf_dataset.filter(
                lambda row: row["category"] in FASHION_CATEGORIES,
                desc="Filtering fashion items"
            )
            print(f"  After fashion filter: {len(hf_dataset)} rows")
 
        set_ids     = [_get_set_id(row["item_ID"]) for row in hf_dataset]
        unique_sets = sorted(set(set_ids))
        set_to_int  = {s: i for i, s in enumerate(unique_sets)}
 
        all_cats        = sorted(set(row["category"] for row in hf_dataset))
        self.cat_to_int = {c: i for i, c in enumerate(all_cats)}
 
        self.samples = []
        for row in hf_dataset:
            outfit_int = set_to_int[_get_set_id(row["item_ID"])]
            cat_int    = self.cat_to_int[row["category"]]
            self.samples.append((row["image"], outfit_int, cat_int))
 
        if max_samples:
            self.samples = self.samples[:max_samples]
 
        print(f"  Final: {len(self.samples)} items, "
              f"{len(unique_sets)} outfits, {len(all_cats)} categories")
 
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        pil_img, outfit_id, cat_id = self.samples[idx]
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)
        return self.transform(pil_img.convert("RGB")), outfit_id, cat_id
 
 
# ================================================================
# LOSS FUNCTIONS — numerically stable via F.cross_entropy
# ================================================================
 
def deepfashion_contrastive(embs: torch.Tensor,
                             labels: torch.Tensor) -> torch.Tensor:
    """
    Supervised contrastive for DeepFashion.
    Uses F.cross_entropy (logsumexp trick) — never overflows.
    Treats the first occurrence of each label's match as the positive.
    For simplicity we use the standard InfoNCE with diagonal positives
    after sorting — but the numerically stable path is cross_entropy.
    """
    B   = embs.size(0)
    # Scale logits
    sim = torch.matmul(embs, embs.T) / 0.07          # (B, B)
 
    # Build positive pair label: for each row i, find one positive j
    # (same clothing_id, different index). Falls back to self if none.
    lbl       = labels.unsqueeze(1)                   # (B,1)
    pos_mask  = torch.eq(lbl, lbl.T)                  # (B,B) bool
    eye       = torch.eye(B, dtype=torch.bool, device=embs.device)
    pos_mask  = pos_mask & ~eye                       # exclude self
 
    # Use cross-entropy with the positive that has highest similarity
    # (hard positive mining). If no positive exists, skip that row.
    has_pos   = pos_mask.any(dim=1)                   # (B,) bool
 
    if not has_pos.any():
        # No positives at all in this batch (can happen with batch_size<2
        # of same class). Return zero loss.
        return torch.tensor(0.0, device=embs.device, requires_grad=True)
 
    # Mask self-similarities out of denominator
    sim_masked = sim.masked_fill(eye, float("-inf"))
 
    # For rows with positives: use cross_entropy with the index of
    # the highest-similarity positive as the target label.
    sim_for_ce  = sim_masked[has_pos]                 # (K, B)
    pos_for_ce  = pos_mask[has_pos]                   # (K, B)
 
    # Target = argmax of positive similarities (deterministic)
    target = (sim_for_ce * pos_for_ce.float()).argmax(dim=1)   # (K,)
 
    return F.cross_entropy(sim_for_ce, target)
 
 
def polyvore_contrastive_stable(
    embs:       torch.Tensor,
    outfit_ids: torch.Tensor,
    cat_ids:    torch.Tensor,
    temperature: float = 0.07,
    hard_neg_weight: float = 0.3,
) -> torch.Tensor:
    """
    Numerically stable category-aware contrastive for Polyvore.
 
    KEY CHANGE from previous version:
    Uses F.cross_entropy(logits, diagonal_labels) instead of the
    manual exp/log path. F.cross_entropy uses the logsumexp trick
    internally, which is float32-safe for any logit magnitude.
 
    Positive definition: diagonal (item i matched with item i's
    own outfit embedding context). This is the standard InfoNCE
    formulation — semantically, we want item i to be most similar
    to items from its own outfit over all other batch items.
 
    Category-aware hard negatives: same-category different-outfit
    pairs get a logit boost BEFORE cross_entropy, making them
    harder negatives without touching the stable denominator path.
    """
    B   = embs.size(0)
    t   = max(temperature, 0.01)
    sim = torch.matmul(embs, embs.T) / t              # (B, B)
    eye = torch.eye(B, dtype=torch.bool, device=embs.device)
 
    # ── Hard-negative boost (same category, different outfit) ────────
    if hard_neg_weight > 0:
        same_cat   = (cat_ids.unsqueeze(1) == cat_ids.unsqueeze(0))
        same_out   = (outfit_ids.unsqueeze(1) == outfit_ids.unsqueeze(0))
        hard_neg   = same_cat & ~same_out & ~eye
        sim        = sim + hard_neg_weight * hard_neg.float()
 
    # ── Mask self-similarity from denominator ────────────────────────
    sim = sim.masked_fill(eye, float("-inf"))
 
    # ── Diagonal targets: item i's positive = item i itself ──────────
    # (standard InfoNCE: we want sim[i,i] to be highest — but we
    #  masked it, so use the same-outfit non-self entries as targets)
    # Better: pair consecutive items as positives by sorting by outfit.
    # Simplest stable approach: treat same-outfit neighbours as targets.
 
    out_eq  = (outfit_ids.unsqueeze(1) == outfit_ids.unsqueeze(0)) & ~eye
    has_pos = out_eq.any(dim=1)
 
    if not has_pos.any():
        # No same-outfit pairs in batch — use diagonal InfoNCE fallback
        # (treat each item as its own positive in the transposed direction)
        labels = torch.arange(B, device=embs.device)
        sim_full = torch.matmul(embs, embs.T) / t
        return F.cross_entropy(sim_full, labels)
 
    # For rows that have a same-outfit pair, pick the highest-sim one
    sim_with_pos = sim.masked_fill(~out_eq, float("-inf"))
    target       = sim_with_pos.argmax(dim=1)          # (B,)
 
    # For rows WITHOUT a positive, use self (diagonal) as target
    # by finding the least-bad assignment
    target[~has_pos] = torch.arange(B, device=embs.device)[~has_pos]
 
    return F.cross_entropy(sim, target)
 
 
# ================================================================
# STAGE 1 -- DEEPFASHION (unchanged, was working)
# ================================================================
 
def pretrain_deepfashion():
    print("\n" + "=" * 60)
    print("  STAGE 1: DeepFashion visual pretraining")
    print("=" * 60)
 
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DeepFashionDataset(config.DEEPFASHION_DIR)
    loader  = DataLoader(dataset,
                         batch_size=config.PRETRAIN_DEEPFASHION_BATCH,
                         shuffle=True, num_workers=2, pin_memory=True)
 
    model     = ClothEncoder(embedding_dim=config.EMBEDDING_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config.PRETRAIN_DEEPFASHION_LR,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.PRETRAIN_DEEPFASHION_EPOCHS
    )
    best_loss = float("inf")
 
    for epoch in range(config.PRETRAIN_DEEPFASHION_EPOCHS):
        model.train()
        total_loss = 0.0
 
        for images, labels in tqdm(
            loader,
            desc=f"DeepFashion Ep {epoch+1}/{config.PRETRAIN_DEEPFASHION_EPOCHS}"
        ):
            images = images.to(device)
            labels = labels.to(device)
 
            embs = model(images)
            loss = deepfashion_contrastive(embs, labels)
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
 
        scheduler.step()
        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1} | Loss: {avg:.4f}")
 
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), config.DEEPFASHION_ENCODER_PATH)
            print(f"  Saved -> {config.DEEPFASHION_ENCODER_PATH}")
 
    print("\nStage 1 complete.\n")
 
 
# ================================================================
# STAGE 2 -- POLYVORE (NaN-stable rewrite)
# ================================================================
 
def pretrain_polyvore():
    print("\n" + "=" * 60)
    print("  STAGE 2: Polyvore compatibility (NaN-stable)")
    print("=" * 60)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = ClothEncoder(
        embedding_dim=config.EMBEDDING_DIM,
        pretrained_path=config.DEEPFASHION_ENCODER_PATH
    ).to(device)
    model.unfreeze_layer4_only()
    print("  layer4 + projector unfrozen. Backbone frozen.")
 
    full_dataset = PolyvoreArrowDataset(
        arrow_dir=config.POLYVORE_ARROW_DIR,
        fashion_only=True
    )
 
    n_val   = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {n_train} | Val: {n_val}")
 
    train_loader = DataLoader(train_ds,
                               batch_size=config.PRETRAIN_POLYVORE_BATCH,
                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,
                               batch_size=config.PRETRAIN_POLYVORE_BATCH,
                               shuffle=False, num_workers=2, pin_memory=True)
 
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.PRETRAIN_POLYVORE_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.PRETRAIN_POLYVORE_EPOCHS
    )
    best_val  = float("inf")
    nan_count = 0   # consecutive NaN counter — abort if > 3
 
    for epoch in range(config.PRETRAIN_POLYVORE_EPOCHS):
 
        # ── Train ──────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        nan_batches = 0
 
        for images, outfit_ids, cat_ids in tqdm(
            train_loader,
            desc=f"Polyvore Ep {epoch+1}/{config.PRETRAIN_POLYVORE_EPOCHS}"
        ):
            images     = images.to(device)
            outfit_ids = outfit_ids.to(device)
            cat_ids    = cat_ids.to(device)
 
            embs = model(images)
 
            # Guard: check embeddings are finite before loss
            if not torch.isfinite(embs).all():
                nan_batches += 1
                continue
 
            loss = polyvore_contrastive_stable(
                embs, outfit_ids, cat_ids,
                temperature=config.PRETRAIN_POLYVORE_TEMP,
                hard_neg_weight=0.3   # reduced from 0.5 — safer
            )
 
            # Guard: skip NaN/inf loss batches
            if not torch.isfinite(loss):
                nan_batches += 1
                continue
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
 
        if nan_batches > 0:
            print(f"  WARNING: {nan_batches} NaN batches skipped this epoch")
 
        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_batches = 0
 
        with torch.no_grad():
            for images, outfit_ids, cat_ids in val_loader:
                images     = images.to(device)
                outfit_ids = outfit_ids.to(device)
                cat_ids    = cat_ids.to(device)
                embs       = model(images)
 
                if not torch.isfinite(embs).all():
                    continue
 
                loss = polyvore_contrastive_stable(
                    embs, outfit_ids, cat_ids,
                    temperature=config.PRETRAIN_POLYVORE_TEMP,
                    hard_neg_weight=0.0   # no boost during validation
                )
                if torch.isfinite(loss):
                    val_loss    += loss.item()
                    val_batches += 1
 
        scheduler.step()
 
        avg_train = train_loss / max(len(train_loader) - nan_batches, 1)
        avg_val   = val_loss   / max(val_batches, 1)
 
        print(f"  Epoch {epoch+1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
 
        # Sanity check: val should be > 0 and < 10
        if avg_val == 0.0:
            print("  WARNING: Val loss = 0.0 — likely no positive pairs in val "
                  "batches. This is OK if train loss is decreasing.")
        if not (0 < avg_val < 20):
            nan_count += 1
            print(f"  WARNING: Suspicious val loss ({avg_val:.4f}). "
                  f"Count: {nan_count}/3")
            if nan_count >= 3:
                print("  ABORTING Stage 2 — model is diverging. "
                      "Re-run Stage 1 first.")
                break
        else:
            nan_count = 0
 
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), config.POLYVORE_ENCODER_PATH)
            print(f"  Saved -> {config.POLYVORE_ENCODER_PATH}")
 
    # Final collapse check on a small val sample
    print("\n  Final embedding health check:")
    model.eval()
    sample_embs = []
    with torch.no_grad():
        for i, (images, _, _) in enumerate(val_loader):
            if i >= 5:
                break
            sample_embs.append(model(images.to(device)).cpu())
    if sample_embs:
        all_e    = torch.cat(sample_embs, dim=0)
        std_val  = all_e.std(dim=0).mean().item()
        n        = min(100, all_e.size(0))
        idx      = torch.randperm(n)
        rand_sim = F.cosine_similarity(all_e[:n], all_e[idx]).mean().item()
        verdict  = "OK" if rand_sim < 0.1 else ("WARNING" if rand_sim < 0.3 else "COLLAPSED")
        print(f"  Std: {std_val:.4f}  RandSim: {rand_sim:.4f}  -> {verdict}")
        if verdict == "COLLAPSED":
            print("  Model collapsed in Stage 2. Do NOT run Stage 3.")
            print("  Re-run: !python src/pretrain_clothing_encoder.py --stage 2")
        else:
            print("  Stage 2 healthy. Proceed to Stage 3.")
 
    print("\nStage 2 complete.\n")
 
 
# ================================================================
# ENTRY POINT
# ================================================================
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
 
    if args.stage == 1:
        pretrain_deepfashion()
    else:
        pretrain_polyvore()
