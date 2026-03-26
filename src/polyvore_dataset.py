import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
 
import config
 
 
# ── Fashion-only categories to keep ────────────────────────────────
# Derived from the actual category values in Marqo/polyvore.
# Non-fashion items (Food, Toys, Furniture, etc.) are excluded so
# the encoder learns garment compatibility, not random object proximity.
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
    """Extract set/outfit ID from item_ID.  '100002074_1' → '100002074'"""
    return item_id.rsplit("_", 1)[0]
 
 
# ================================================================
# MAIN DATASET CLASS
# ================================================================
 
class PolyvoreArrowDataset(Dataset):
    """
    Loads the Polyvore dataset from HuggingFace Arrow shards.
 
    Each __getitem__ returns:
        img        : (3, 224, 224) tensor
        outfit_id  : int  — items sharing outfit_id are compatible
 
    This is designed for use with InfoNCELoss / supervised contrastive:
    the loss treats same-outfit items as positives, cross-outfit as negatives.
 
    Usage:
        dataset = PolyvoreArrowDataset(config.POLYVORE_ARROW_DIR)
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)
        for imgs, outfit_ids in loader:
            ...
    """
 
    def __init__(self,
                 arrow_dir: str = config.POLYVORE_ARROW_DIR,
                 fashion_only: bool = True,
                 max_samples: int = None):
        """
        arrow_dir     : path to the "data" folder containing
                        dataset_info.json + .arrow shards
        fashion_only  : if True, filter out non-clothing items
        max_samples   : cap dataset size (useful for quick runs)
        """
        from datasets import load_from_disk
 
        print(f"  Loading Polyvore from: {arrow_dir}")
        hf_dataset = load_from_disk(arrow_dir)
        print(f"  Raw rows: {len(hf_dataset)}")
 
        # ── Filter to fashion categories ────────────────────────────
        if fashion_only:
            hf_dataset = hf_dataset.filter(
                lambda row: row["category"] in FASHION_CATEGORIES,
                desc="Filtering fashion items"
            )
            print(f"  After fashion filter: {len(hf_dataset)} rows")
 
        # ── Build outfit_id → integer mapping ───────────────────────
        # item_ID = "{set_id}_{item_index}" → group by set_id
        set_ids     = [_get_set_id(row["item_ID"]) for row in hf_dataset]
        unique_sets = sorted(set(set_ids))
        self.set_id_to_int = {s: i for i, s in enumerate(unique_sets)}
 
        # ── Store as list of (pil_image, outfit_id_int) ─────────────
        self.samples = []
        for i, row in enumerate(hf_dataset):
            outfit_int = self.set_id_to_int[_get_set_id(row["item_ID"])]
            self.samples.append((row["image"], outfit_int, row["category"]))
 
        if max_samples:
            self.samples = self.samples[:max_samples]
 
        print(f"  Final dataset : {len(self.samples)} items, "
              f"{len(unique_sets)} outfits")
 
        # ── Transform ───────────────────────────────────────────────
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
        pil_img, outfit_id, category = self.samples[idx]
 
        # HuggingFace Image column returns a PIL Image directly
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)
        pil_img = pil_img.convert("RGB")
 
        return self.transform(pil_img), outfit_id, category
 
 
# ================================================================
# TRIPLET DATASET  (alternative — returns anchor/pos/neg explicitly)
# ================================================================
 
class PolyvoreTripletDataset(Dataset):
    """
    Returns (anchor, positive, negative) image triplets.
 
    anchor   + positive : same outfit (compatible)
    negative            : different outfit
 
    explicit triplet training over
    in-batch negatives (InfoNCE).
    """
 
    def __init__(self,
                 arrow_dir: str = config.POLYVORE_ARROW_DIR,
                 fashion_only: bool = True):
        from datasets import load_from_disk
 
        print(f"  Loading Polyvore triplets from: {arrow_dir}")
        hf_dataset = load_from_disk(arrow_dir)
 
        if fashion_only:
            hf_dataset = hf_dataset.filter(
                lambda row: row["category"] in FASHION_CATEGORIES,
                desc="Filtering fashion items"
            )
 
        # Group by outfit (set_id)
        outfit_to_items = defaultdict(list)
        for row in hf_dataset:
            set_id = _get_set_id(row["item_ID"])
            outfit_to_items[set_id].append(row["image"])
 
        # Keep only outfits with >= 2 items (need at least anchor + positive)
        self.outfits = {k: v for k, v in outfit_to_items.items()
                        if len(v) >= 2}
        self.outfit_keys = list(self.outfits.keys())
 
        # Flat item list for negative sampling
        self.all_images = [img for imgs in self.outfits.values()
                           for img in imgs]
 
        print(f"  Triplet dataset: {len(self.outfit_keys)} outfits, "
              f"{len(self.all_images)} total items")
 
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])
 
    def _to_tensor(self, pil_img):
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)
        return self.transform(pil_img.convert("RGB"))
 
    def __len__(self):
        return len(self.outfit_keys)
 
    def __getitem__(self, idx):
        key    = self.outfit_keys[idx]
        items  = self.outfits[key]
        anchor_img, pos_img = random.sample(items, 2)
 
        # Negative: from a different outfit
        outfit_set = set(id(x) for x in items)
        while True:
            neg_img = random.choice(self.all_images)
            if id(neg_img) not in outfit_set:
                break
 
        return {
            "anchor":   self._to_tensor(anchor_img),
            "positive": self._to_tensor(pos_img),
            "negative": self._to_tensor(neg_img),
        }
 
 
# ================================================================
# QUICK DIAGNOSTIC — run this in Colab to verify the data loads
# ================================================================
 
def verify_polyvore(arrow_dir: str = config.POLYVORE_ARROW_DIR,
                    n_samples: int = 5):
    """
    Call this in Colab before training to confirm:
      1. Arrow shards load correctly
      2. item_ID → set_id grouping works
      3. Images decode without errors
      4. outfit_id integers are assigned correctly
 
    Usage in Colab:
        from polyvore_dataset import verify_polyvore
        verify_polyvore()
    """
    print("=" * 55)
    print("  Polyvore Arrow dataset verification")
    print("=" * 55)
 
    dataset = PolyvoreArrowDataset(arrow_dir, fashion_only=True,
                                   max_samples=200)
 
    print(f"\n  Dataset length  : {len(dataset)}")
    print(f"  Sample __getitem__:")
 
    for i in range(min(n_samples, len(dataset))):
        img_tensor, outfit_id, category = dataset[i]
        print(f"    [{i}] shape={img_tensor.shape}  outfit={outfit_id}  category={category}")
 
    # Check that multiple items share outfit IDs (compatibility pairs exist)
    outfit_ids = [dataset[i][1] for i in range(min(100, len(dataset)))]
    from collections import Counter
    counts = Counter(outfit_ids)
    multi  = sum(1 for c in counts.values() if c > 1)
    print(f"\n  Out of first 100 items: {multi} outfits have >1 item "
          f"(positive pairs available)")
 
    print("\n  ✓ Polyvore dataset loads correctly!")
    return dataset
 
 
if __name__ == "__main__":
    verify_polyvore()
