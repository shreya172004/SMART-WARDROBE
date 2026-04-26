# ================================================================
# DATASET — Body + Clothing (Fashionista)
# ================================================================
# Combines:
# ✔ Body feature standardization (NEW)
# ✔ Noise filtering (OLD - important)
# ✔ Consistent transforms (NEW)
# ✔ Optional augmentation
# ================================================================

import os
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import config


# ================================================================
# TRANSFORMS
# ================================================================

def get_train_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN,
                             std=config.IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN,
                             std=config.IMAGENET_STD),
    ])


# ================================================================
# DATASET CLASS
# ================================================================

class BodyClothDataset(Dataset):
    """
    Body ↔ Clothing dataset.

    Features:
    ✔ Body vector standardization (critical)
    ✔ Removes noisy samples (incomplete body detection)
    ✔ Supports augmentation
    ✔ Robust CSV handling (multiple formats)
    """

    def __init__(self,
                 image_dir,
                 body_csv,
                 augment=False,
                 body_completeness_threshold=0.7):

        self.image_dir = image_dir
        self.augment = augment
        self.body_completeness_threshold = body_completeness_threshold

        # Choose transform
        self.transform = get_train_transform() if augment else get_eval_transform()

        # ------------------------------------------------
        # LOAD CSV
        # ------------------------------------------------
        df = pd.read_csv(body_csv)

        # Detect image column automatically
        if "image_name" in df.columns:
            image_col = "image_name"
        elif "filename" in df.columns:
            image_col = "filename"
        elif "image" in df.columns:
            image_col = "image"
        else:
            raise ValueError("No image filename column found in CSV")

        # ------------------------------------------------
        # HANDLE BODY VECTOR FORMAT
        # ------------------------------------------------
        if "body_vector" in df.columns:
            # Format: "[1.2, 0.8, ...]"
            df["body_vector"] = df["body_vector"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

            df = df[df["body_vector"].apply(
                lambda x: isinstance(x, (list, tuple)) and len(x) == config.BODY_INPUT_DIM
            )].copy()

        else:
            # Format: separate columns
            measurement_cols = [c for c in df.columns if c != image_col]

            vals = df[measurement_cols].values.astype(float)

            # Filter incomplete body detections
            completeness = (vals != 0).mean(axis=1)
            df = df[completeness >= body_completeness_threshold].copy()

            df["body_vector"] = df[measurement_cols].values.tolist()

        # ------------------------------------------------
        # FILTER MISSING IMAGES
        # ------------------------------------------------
        df["image_path"] = df[image_col].apply(
            lambda x: os.path.join(image_dir, str(x))
        )

        df = df[df["image_path"].apply(os.path.exists)].copy()
        df = df.reset_index(drop=True)

        self.df = df

        print(f"  Dataset: {len(self.df)} valid samples from {image_dir}")

        # ------------------------------------------------
        # BODY STANDARDIZATION (VERY IMPORTANT)
        # ------------------------------------------------
        body_matrix = torch.tensor(
            self.df["body_vector"].tolist(),
            dtype=torch.float32
        )

        self.body_mean = body_matrix.mean(dim=0)
        self.body_std = body_matrix.std(dim=0)

        # Avoid division by zero
        self.body_std[self.body_std < 1e-8] = 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------------------------
        # IMAGE
        # -------------------------
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        # -------------------------
        # BODY VECTOR
        # -------------------------
        body_vec = torch.tensor(row["body_vector"], dtype=torch.float32)

        # Standardize (IMPORTANT)
        body_vec = (body_vec - self.body_mean) / (self.body_std + 1e-8)

        return {
            "body": body_vec,
            "image": image,
            "image_path": row["image_path"]
        }