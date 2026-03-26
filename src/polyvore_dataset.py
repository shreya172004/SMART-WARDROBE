"""
polyvore_dataset.py  —  Polyvore HuggingFace Arrow format loader
"""

import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict

import config


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


class PolyvoreArrowDataset(Dataset):

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
        self.set_id_to_int = {s: i for i, s in enumerate(unique_sets)}

        all_cats        = sorted(set(row["category"] for row in hf_dataset))
        self.cat_to_int = {c: i for i, c in enumerate(all_cats)}

        self.samples = []
        for i, row in enumerate(hf_dataset):
            outfit_int = self.set_id_to_int[_get_set_id(row["item_ID"])]
            cat_int    = self.cat_to_int[row["category"]]
            self.samples.append((row["image"], outfit_int, cat_int))

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"  Final dataset : {len(self.samples)} items, {len(unique_sets)} outfits")

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
        pil_img, outfit_id, cat_id = self.samples[idx]   # 3 values
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)
        return self.transform(pil_img.convert("RGB")), outfit_id, cat_id  # 3 values


class PolyvoreTripletDataset(Dataset):

    def __init__(self, arrow_dir=config.POLYVORE_ARROW_DIR, fashion_only=True):
        from datasets import load_from_disk
        hf_dataset = load_from_disk(arrow_dir)
        if fashion_only:
            hf_dataset = hf_dataset.filter(
                lambda row: row["category"] in FASHION_CATEGORIES,
                desc="Filtering fashion items"
            )
        outfit_to_items = defaultdict(list)
        for row in hf_dataset:
            outfit_to_items[_get_set_id(row["item_ID"])].append(row["image"])
        self.outfits     = {k: v for k, v in outfit_to_items.items() if len(v) >= 2}
        self.outfit_keys = list(self.outfits.keys())
        self.all_images  = [img for imgs in self.outfits.values() for img in imgs]
        self.transform   = transforms.Compose([
            transforms.Resize((256, 256)), transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _to_tensor(self, pil_img):
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)
        return self.transform(pil_img.convert("RGB"))

    def __len__(self):
        return len(self.outfit_keys)

    def __getitem__(self, idx):
        items = self.outfits[self.outfit_keys[idx]]
        anchor_img, pos_img = random.sample(items, 2)   # PIL images, not tuples
        outfit_set = set(id(x) for x in items)
        while True:
            neg_img = random.choice(self.all_images)
            if id(neg_img) not in outfit_set:
                break
        return {"anchor": self._to_tensor(anchor_img),
                "positive": self._to_tensor(pos_img),
                "negative": self._to_tensor(neg_img)}


def verify_polyvore(arrow_dir=config.POLYVORE_ARROW_DIR, n_samples=5):
    print("=" * 55)
    print("  Polyvore Arrow dataset verification")
    print("=" * 55)
    dataset = PolyvoreArrowDataset(arrow_dir, fashion_only=True, max_samples=200)
    print(f"\n  Dataset length  : {len(dataset)}")
    print(f"  Sample __getitem__:")
    for i in range(min(n_samples, len(dataset))):
        img_tensor, outfit_id, cat_id = dataset[i]
        print(f"    [{i}] shape={img_tensor.shape}  outfit={outfit_id}  cat_id={cat_id}")
    from collections import Counter
    counts = Counter(dataset[i][1] for i in range(min(100, len(dataset))))
    multi  = sum(1 for c in counts.values() if c > 1)
    print(f"\n  Outfits with >1 item in first 100: {multi}  (positive pairs available)")
    print("\n  Polyvore dataset loads correctly!")
    return dataset


if __name__ == "__main__":
    verify_polyvore()