import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
 
import config
from cloth_encoder import cloth_transform, upper_body_transform
 
 
class BodyClothDataset(Dataset):
    """
    Fashionista body-clothing pairs dataset.
 
    Changes from v1:
    - Optional upper_body_crop (default True) to reduce pose/shoe bias
    - body_completeness_threshold: filters out rows where too many
      body measurements are zero (noisy / failed pose detection)
    - augmentation: random horizontal flip during training
    """
 
    def __init__(self,
                 image_dir,
                 body_csv,
                 use_upper_crop: bool = True,
                 augment: bool = False,
                 body_completeness_threshold: float = 0.7):
 
        self.image_dir = image_dir
        self.use_upper_crop = use_upper_crop
 
        body_df = pd.read_csv(body_csv)
 
        # ── Filter: only rows whose image files exist ────────────────
        valid_names = [
            img for img in body_df["image"].tolist()
            if os.path.exists(os.path.join(image_dir, img))
        ]
        body_df = body_df[body_df["image"].isin(valid_names)].reset_index(drop=True)
 
        # ── Filter: remove rows with mostly-zero body vectors ────────
        # These come from failed MediaPipe pose detection and act as noise
        measurement_cols = [c for c in body_df.columns if c != "image"]
        vals = body_df[measurement_cols].values.astype(float)
        completeness = (vals != 0).mean(axis=1)
        body_df = body_df[completeness >= body_completeness_threshold].reset_index(drop=True)
 
        self.body_df = body_df
        print(f"  Dataset: {len(self.body_df)} valid samples from {image_dir}")
 
        # ── Transforms ───────────────────────────────────────────────
        if augment:
            self.augment_tf = transforms.RandomHorizontalFlip(p=0.5)
        else:
            self.augment_tf = None
 
    def __len__(self):
        return len(self.body_df)
 
    def __getitem__(self, idx):
        row      = self.body_df.iloc[idx]
        img_name = row["image"]
        img_path = os.path.join(self.image_dir, img_name)
 
        # Body vector (raw — passed to loss for debiasing)
        body_vector = torch.tensor(
            row.drop("image").to_numpy(dtype=float),
            dtype=torch.float32
        )
 
        # Clothing image
        img = Image.open(img_path).convert("RGB")
        if self.augment_tf is not None:
            img = self.augment_tf(img)
 
        if self.use_upper_crop:
            img_tensor = upper_body_transform(img)
        else:
            img_tensor = cloth_transform(img)
 
        return {
            "image": img_tensor,
            "body":  body_vector
        }
