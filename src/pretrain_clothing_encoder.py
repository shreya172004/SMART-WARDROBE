"""
pretrain_clothing_encoder.py
 
Two-stage clothing encoder pretraining:
  Stage 1 -> DeepFashion -> learn visual garment features
  Stage 2 -> Polyvore    -> learn outfit compatibility (category-aware)
 
BUGS FIXED vs the "category-aware" version:
-----------------------------------------------------------------
Bug 1 (crash): DataLoader loop unpacked 3 values but dataset only
               returned 2. Fixed: dataset now stores & returns
               category_id as a third value.
 
Bug 2 (silent corruption): hash(str) is randomised per process.
               On DataLoader num_workers>0 workers get different
               hash values for the same category string -> mask wrong.
               Fixed: deterministic cat_to_int lookup built at init.
 
Bug 3 (loss collapse): Restricting denominator to same-category only
               gives ~1-2 negatives per row in a batch of 32 ->
               denominator near-zero -> loss explodes/collapses.
               Fixed: full denominator (all non-self pairs) + logit
               boost on hard (same-category) negatives instead.
 
Bug 4 (data): category was never stored in self.samples, so even
               fixing the loop would crash. Fixed: tuple extended
               to (img, outfit_id_int, category_id_int).
-----------------------------------------------------------------
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
# FASHION CATEGORIES FILTER
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
    """'100002074_1' -> '100002074'"""
    return item_id.rsplit("_", 1)[0]
 
 
# ================================================================
# POLYVORE DATASET  (returns 3 values: img, outfit_id, category_id)
# ================================================================
 
class PolyvoreArrowDataset(Dataset):
    """
    Loads Polyvore from HuggingFace Arrow shards via load_from_disk().
 
    __getitem__ returns:
        img_tensor   : (3, 224, 224) float32
        outfit_id    : int  same outfit_id = compatible pair (positive)
        category_id  : int  deterministic int, NOT hash() - see Bug 2 fix
    """
 
    def __init__(self,
                 arrow_dir=config.POLYVORE_ARROW_DIR,
                 fashion_only=True,
                 max_samples=None):
 
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
 
        # Deterministic outfit_id -> int
        set_ids     = [_get_set_id(row["item_ID"]) for row in hf_dataset]
        unique_sets = sorted(set(set_ids))
        set_to_int  = {s: i for i, s in enumerate(unique_sets)}
 
        # Deterministic category -> int (FIX Bug 2: no hash())
        # Built once at init; all DataLoader workers inherit via pickle.
        all_cats        = sorted(set(row["category"] for row in hf_dataset))
        self.cat_to_int = {c: i for i, c in enumerate(all_cats)}
 
        # Store (pil_image, outfit_id_int, category_id_int)  FIX Bug 4
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
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
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
# LOSS FUNCTIONS
# ================================================================
 
def deepfashion_contrastive(embs, labels):
    """Standard supervised contrastive. Same clothing_id = positive."""
    B   = embs.size(0)
    sim = torch.matmul(embs, embs.T) / 0.07
    eye = torch.eye(B, device=embs.device)
 
    exp_sim   = torch.exp(sim) * (1 - eye)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    log_prob  = sim - log_denom
 
    lbl      = labels.unsqueeze(1)
    pos_mask = torch.eq(lbl, lbl.T).float() * (1 - eye)
    n_pos    = pos_mask.sum(dim=1).clamp(min=1)
    return (-(pos_mask * log_prob).sum(dim=1) / n_pos).mean()
 
 
def polyvore_contrastive_category_aware(
    embs, outfit_ids, cat_ids,
    temperature=0.07,
    hard_neg_weight=0.5
):
    """
    Category-aware contrastive for Polyvore.
 
    Positives  : same outfit_id
    Denominator: ALL non-self pairs (FIX Bug 3 - keeps denominator stable)
    Hard negatives: same-category negatives get a logit boost of
                    hard_neg_weight so the model works harder to push
                    apart items from the same category that are NOT
                    in the same outfit -- without shrinking the denom.
    """
    B   = embs.size(0)
    t   = max(temperature, 0.01)
    sim = torch.matmul(embs, embs.T) / t
    eye = torch.eye(B, device=embs.device)
 
    # Same-category mask (int == int, fully deterministic)
    cat_mask = (cat_ids.unsqueeze(1) == cat_ids.unsqueeze(0)).float()
 
    # Positive mask (same outfit)
    out_mask = (outfit_ids.unsqueeze(1) == outfit_ids.unsqueeze(0)).float()
 
    # Hard negatives: same category, different outfit, not self
    hard_neg = cat_mask * (1 - out_mask) * (1 - eye)
 
    # Boost hard negatives in logit space
    sim_boosted = sim + hard_neg_weight * hard_neg
 
    # Full denominator (all non-self)
    exp_sim   = torch.exp(sim_boosted) * (1 - eye)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    log_prob  = sim_boosted - log_denom
 
    # Loss over positive pairs
    pos_off = out_mask * (1 - eye)
    n_pos   = pos_off.sum(dim=1).clamp(min=1)
    return (-(pos_off * log_prob).sum(dim=1) / n_pos).mean()
 
 
# ================================================================
# STAGE 1 -- DEEPFASHION
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
# STAGE 2 -- POLYVORE (CATEGORY-AWARE, ALL BUGS FIXED)
# ================================================================
 
def pretrain_polyvore():
    print("\n" + "=" * 60)
    print("  STAGE 2: Polyvore compatibility (category-aware)")
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
    best_val = float("inf")
 
    for epoch in range(config.PRETRAIN_POLYVORE_EPOCHS):
 
        # Train
        model.train()
        train_loss = 0.0
 
        for images, outfit_ids, cat_ids in tqdm(    # FIX Bug 1: 3 values
            train_loader,
            desc=f"Polyvore Ep {epoch+1}/{config.PRETRAIN_POLYVORE_EPOCHS}"
        ):
            images     = images.to(device)
            outfit_ids = outfit_ids.to(device)
            cat_ids    = cat_ids.to(device)          # FIX Bug 2: int tensor
 
            embs = model(images)
 
            loss = polyvore_contrastive_category_aware(  # FIX Bug 3
                embs, outfit_ids, cat_ids,
                temperature=config.PRETRAIN_POLYVORE_TEMP,
                hard_neg_weight=0.5
            )
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
 
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, outfit_ids, cat_ids in val_loader:
                images     = images.to(device)
                outfit_ids = outfit_ids.to(device)
                cat_ids    = cat_ids.to(device)
                embs       = model(images)
                val_loss  += polyvore_contrastive_category_aware(
                    embs, outfit_ids, cat_ids,
                    temperature=config.PRETRAIN_POLYVORE_TEMP
                ).item()
 
        scheduler.step()
        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"  Epoch {epoch+1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
 
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), config.POLYVORE_ENCODER_PATH)
            print(f"  Saved -> {config.POLYVORE_ENCODER_PATH}")
 
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
 
