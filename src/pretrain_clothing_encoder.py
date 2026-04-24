"""
pretrain_clothing_encoder.py  —  Final stable version
 
FIXES IN THIS VERSION:
──────────────────────────────────────────────────────────────────
Fix 1  HuggingFace filter cache corruption
   The two cache-*.arrow files (445 KB each) in your Drive folder
   are corrupt filter caches from a previous failed run. They cause
   load_from_disk+filter to return almost no data silently.
   Fix: load raw Arrow shards manually with pyarrow, bypassing the
   HuggingFace cache system entirely. No cache files are written.
 
Fix 2  Val loss = 0.0000 (no positive pairs in val batches)
   With 19,268 outfits and batch_size=32, a random 90/10 split gives
   P(same-outfit pair in batch) ≈ 0.002 per batch. Almost every val
   batch has zero positive pairs → loss = 0 from the clamp.
   Fix: outfit-aware train/val split. We split by outfit set_id,
   so all items from a given outfit stay together in train OR val.
   Val outfits are sampled to guarantee ≥2 items per outfit → every
   val batch has real positive pairs → real loss values.
 
Fix 3  Numerically stable loss via F.cross_entropy (kept from prev fix)
──────────────────────────────────────────────────────────────────
"""
 
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import random
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
# POLYVORE DATASET — loads via pyarrow directly (no HF cache)
# ================================================================
 
class PolyvoreArrowDataset(Dataset):
    """
    Loads Polyvore Arrow shards directly via pyarrow.
    Bypasses HuggingFace datasets cache completely — no cache-*.arrow
    files are written or read. This avoids the corrupt cache problem.
 
    __getitem__ returns (img_tensor, outfit_id_int, category_id_int).
    """
 
    def __init__(self, arrow_dir=config.POLYVORE_ARROW_DIR,
                 fashion_only=True, indices=None):
        """
        arrow_dir    : folder containing data-00000-of-00006.arrow etc.
        fashion_only : filter to FASHION_CATEGORIES only
        indices      : optional list of row indices to use (for splits)
        """
        import pyarrow as pa
        import glob, os
 
        print(f"  Loading Polyvore shards from: {arrow_dir}")
 
        # Find all data shards (not cache files)
        shard_paths = sorted(glob.glob(
            os.path.join(arrow_dir, "data-*-of-*.arrow")
        ))
        if not shard_paths:
            raise FileNotFoundError(
                f"No data-*-of-*.arrow files found in {arrow_dir}.\n"
                f"Files present: {os.listdir(arrow_dir)}"
            )
        print(f"  Found {len(shard_paths)} shards")
 
        # Read all shards into memory as a list of dicts
        raw_rows = []
        for path in shard_paths:
            reader = pa.ipc.open_file(path)
            table  = reader.read_all()
            # Convert to Python dicts row by row
            n = table.num_rows
            # Get column arrays
            images    = table.column("image")
            categories = table.column("category")
            item_ids   = table.column("item_ID")
 
            for i in range(n):
                cat = categories[i].as_py()
                if fashion_only and cat not in FASHION_CATEGORIES:
                    continue
                # Image is stored as a dict {"bytes": b"...", "path": ...}
                # or directly as bytes depending on HF version
                img_val = images[i].as_py()
                raw_rows.append({
                    "image":    img_val,
                    "category": cat,
                    "item_ID":  item_ids[i].as_py(),
                })
 
        print(f"  Rows after fashion filter: {len(raw_rows)}")
 
        # Apply index subset if provided (for train/val split)
        if indices is not None:
            raw_rows = [raw_rows[i] for i in indices]
 
        # Build deterministic mappings
        set_ids     = [_get_set_id(r["item_ID"]) for r in raw_rows]
        unique_sets = sorted(set(set_ids))
        set_to_int  = {s: i for i, s in enumerate(unique_sets)}
 
        all_cats        = sorted(set(r["category"] for r in raw_rows))
        self.cat_to_int = {c: i for i, c in enumerate(all_cats)}
 
        # Store samples
        self.samples = []
        for r in raw_rows:
            outfit_int = set_to_int[_get_set_id(r["item_ID"])]
            cat_int    = self.cat_to_int[r["category"]]
            self.samples.append((r["image"], outfit_int, cat_int))
 
        print(f"  Dataset: {len(self.samples)} items, "
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
 
    def _decode_image(self, img_val):
        """Handle both dict {'bytes':...} and raw bytes formats."""
        import io
        if isinstance(img_val, dict):
            raw = img_val.get("bytes") or img_val.get("path")
            if isinstance(raw, bytes):
                return Image.open(io.BytesIO(raw)).convert("RGB")
            elif isinstance(raw, str):
                return Image.open(raw).convert("RGB")
        elif isinstance(img_val, bytes):
            return Image.open(io.BytesIO(img_val)).convert("RGB")
        elif isinstance(img_val, Image.Image):
            return img_val.convert("RGB")
        raise ValueError(f"Unknown image format: {type(img_val)}")
 
    def __getitem__(self, idx):
        img_val, outfit_id, cat_id = self.samples[idx]
        pil_img = self._decode_image(img_val)
        return self.transform(pil_img), outfit_id, cat_id
 
 
def make_outfit_aware_split(dataset: PolyvoreArrowDataset,
                             val_fraction: float = 0.1,
                             min_val_outfit_size: int = 2,
                             seed: int = 42):
    """
    Split dataset into train/val by outfit (set_id), not by row.
 
    Guarantees:
    - All items from a given outfit stay together (train OR val)
    - Val outfits all have ≥ min_val_outfit_size items
      → every val batch has real positive pairs
    - Roughly val_fraction of total items end up in val
 
    Returns: (train_indices, val_indices) — lists of int
    """
    rng = random.Random(seed)
 
    # Group sample indices by outfit_id
    outfit_to_indices = defaultdict(list)
    for i, (_, outfit_id, _) in enumerate(dataset.samples):
        outfit_to_indices[outfit_id].append(i)
 
    # Only outfits with ≥ min_val_outfit_size are eligible for val
    eligible_val = {oid: idxs for oid, idxs in outfit_to_indices.items()
                    if len(idxs) >= min_val_outfit_size}
    ineligible   = {oid: idxs for oid, idxs in outfit_to_indices.items()
                    if len(idxs) < min_val_outfit_size}
 
    n_total   = len(dataset)
    n_val_target = int(n_total * val_fraction)
 
    # Shuffle eligible outfits and pick until we reach val target
    eligible_list = list(eligible_val.keys())
    rng.shuffle(eligible_list)
 
    val_outfit_ids  = set()
    val_count       = 0
    for oid in eligible_list:
        if val_count >= n_val_target:
            break
        val_outfit_ids.add(oid)
        val_count += len(eligible_val[oid])
 
    # Build index lists
    val_indices   = [i for oid in val_outfit_ids
                     for i in outfit_to_indices[oid]]
    train_indices = [i for oid, idxs in outfit_to_indices.items()
                     if oid not in val_outfit_ids
                     for i in idxs]
 
    print(f"  Outfit-aware split:")
    print(f"    Train: {len(train_indices)} items, "
          f"{len(outfit_to_indices) - len(val_outfit_ids)} outfits")
    print(f"    Val:   {len(val_indices)} items, "
          f"{len(val_outfit_ids)} outfits "
          f"(all have ≥{min_val_outfit_size} items → positive pairs guaranteed)")
 
    return train_indices, val_indices
 
 
# ================================================================
# LOSS FUNCTIONS — numerically stable via F.cross_entropy
# ================================================================
 
def deepfashion_contrastive(embs: torch.Tensor,
                             labels: torch.Tensor) -> torch.Tensor:
    """Standard supervised contrastive using F.cross_entropy (stable)."""
    B    = embs.size(0)
    sim  = torch.matmul(embs, embs.T) / 0.07
    eye  = torch.eye(B, dtype=torch.bool, device=embs.device)
 
    lbl      = labels.unsqueeze(1)
    pos_mask = torch.eq(lbl, lbl.T) & ~eye
 
    has_pos  = pos_mask.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=embs.device, requires_grad=True)
 
    sim_masked   = sim.masked_fill(eye, float("-inf"))
    sim_for_ce   = sim_masked[has_pos]
    pos_for_ce   = pos_mask[has_pos]
    target       = (sim_for_ce * pos_for_ce.float()).argmax(dim=1)
 
    return F.cross_entropy(sim_for_ce, target)
 
 
def polyvore_contrastive_stable(
    embs:            torch.Tensor,
    outfit_ids:      torch.Tensor,
    cat_ids:         torch.Tensor,
    temperature:     float = 0.07,
    hard_neg_weight: float = 0.3,
) -> torch.Tensor:
    """
    Numerically stable contrastive loss for Polyvore.
    Uses F.cross_entropy (logsumexp trick) — never overflows.
    Positive = highest-similarity same-outfit item.
    Hard negatives = same-category different-outfit items get logit boost.
    """
    B   = embs.size(0)
    t   = max(temperature, 0.01)
    sim = torch.matmul(embs, embs.T) / t
    eye = torch.eye(B, dtype=torch.bool, device=embs.device)
 
    # Category-aware hard negative boost
    if hard_neg_weight > 0:
        same_cat = (cat_ids.unsqueeze(1) == cat_ids.unsqueeze(0))
        same_out = (outfit_ids.unsqueeze(1) == outfit_ids.unsqueeze(0))
        hard_neg = same_cat & ~same_out & ~eye
        sim      = sim + hard_neg_weight * hard_neg.float()
 
    # Mask self from denominator
    sim_masked = sim.masked_fill(eye, float("-inf"))
 
    # Find positive (same-outfit, highest similarity)
    out_eq  = (outfit_ids.unsqueeze(1) == outfit_ids.unsqueeze(0)) & ~eye
    has_pos = out_eq.any(dim=1)
 
    if not has_pos.any():
        # Fallback: no same-outfit pair in batch — use standard InfoNCE
        # (happens only in val if batch is unlucky — not in train with
        # outfit-aware split ensuring same-outfit items cluster together)
        labels = torch.arange(B, device=embs.device)
        sim_full = torch.matmul(embs, embs.T) / t
        return F.cross_entropy(sim_full, labels)
 
    # For each row: target = index of highest-sim same-outfit item
    sim_pos = sim_masked.masked_fill(~out_eq, float("-inf"))
    target  = sim_pos.argmax(dim=1)
 
    # Rows without a same-outfit pair: use nearest non-self item
    fallback = sim_masked.argmax(dim=1)
    target   = torch.where(has_pos, target, fallback)
 
    return F.cross_entropy(sim_masked, target)
 
 
# ================================================================
# STAGE 1 -- DEEPFASHION (unchanged — was working correctly)
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
            embs   = model(images)
            loss   = deepfashion_contrastive(embs, labels)
 
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
# STAGE 2 -- POLYVORE  (cache-free + outfit-aware val split)
# ================================================================
 
def pretrain_polyvore():
    print("\n" + "=" * 60)
    print("  STAGE 2: Polyvore compatibility")
    print("=" * 60)
    print("  NOTE: Loading via pyarrow directly — no HF cache used.")
    print("  If you see cache-*.arrow files in your Drive, they are")
    print("  from a previous run and can be safely deleted.\n")
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = ClothEncoder(
        embedding_dim=config.EMBEDDING_DIM,
        pretrained_path=config.DEEPFASHION_ENCODER_PATH
    ).to(device)
    model.unfreeze_layer4_only()
    print("  layer4 + projector unfrozen. Backbone frozen.\n")
 
    # Load full dataset (no HF cache)
    full_dataset = PolyvoreArrowDataset(
        arrow_dir=config.POLYVORE_ARROW_DIR,
        fashion_only=True
    )
 
    # Outfit-aware split: guarantees positive pairs in val batches
    train_idx, val_idx = make_outfit_aware_split(
        full_dataset, val_fraction=0.1, min_val_outfit_size=2, seed=42
    )
 
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=config.PRETRAIN_POLYVORE_BATCH,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=config.PRETRAIN_POLYVORE_BATCH,
        shuffle=False, num_workers=2, pin_memory=True
    )
 
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.PRETRAIN_POLYVORE_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.PRETRAIN_POLYVORE_EPOCHS
    )
    best_val = float("inf")
 
    for epoch in range(config.PRETRAIN_POLYVORE_EPOCHS):
 
        # ── Train ──────────────────────────────────────────────────
        model.train()
        train_loss  = 0.0
        nan_batches = 0
 
        for images, outfit_ids, cat_ids in tqdm(
            train_loader,
            desc=f"Polyvore Ep {epoch+1}/{config.PRETRAIN_POLYVORE_EPOCHS}"
        ):
            images     = images.to(device)
            outfit_ids = outfit_ids.to(device)
            cat_ids    = cat_ids.to(device)
            embs       = model(images)
 
            if not torch.isfinite(embs).all():
                nan_batches += 1
                continue
 
            loss = polyvore_contrastive_stable(
                embs, outfit_ids, cat_ids,
                temperature=config.PRETRAIN_POLYVORE_TEMP,
                hard_neg_weight=0.3
            )
 
            if not torch.isfinite(loss):
                nan_batches += 1
                continue
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
 
        if nan_batches > 0:
            print(f"  WARNING: {nan_batches} NaN batches skipped")
 
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
                    hard_neg_weight=0.0
                )
                if torch.isfinite(loss):
                    val_loss    += loss.item()
                    val_batches += 1
 
        scheduler.step()
 
        n_train_batches = max(len(train_loader) - nan_batches, 1)
        avg_train = train_loss / n_train_batches
        avg_val   = val_loss   / max(val_batches, 1)
 
        print(f"  Epoch {epoch+1} | Train: {avg_train:.4f} | "
              f"Val: {avg_val:.4f}  [{val_batches} val batches]")
 
        # Val should be a real loss value (3–5 range for InfoNCE at start)
        if avg_val == 0.0 and val_batches == 0:
            print("  WARNING: No finite val batches — check data loading.")
 
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), config.POLYVORE_ENCODER_PATH)
            print(f"  Saved -> {config.POLYVORE_ENCODER_PATH}")
 
    # ── Collapse check ─────────────────────────────────────────────
    print("\n  Embedding health check:")
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
        verdict  = ("OK" if rand_sim < 0.1
                    else "WARNING" if rand_sim < 0.3
                    else "COLLAPSED")
        print(f"  Std: {std_val:.4f}  RandSim: {rand_sim:.4f}  -> {verdict}")
        if verdict == "COLLAPSED":
            print("  STOP: Model collapsed. Do NOT run Stage 3.")
            print("  Action: delete cache-*.arrow from Drive, re-run Stage 2.")
        else:
            print("  Stage 2 healthy. Proceed to: !python src/train.py")
 
    print("\nStage 2 complete.\n")
 
 
# ================================================================
# ENTRY POINT
# ================================================================
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                        help="1=DeepFashion, 2=Polyvore")
    args = parser.parse_args()
 
    if args.stage == 1:
        pretrain_deepfashion()
    else:
        pretrain_polyvore()
