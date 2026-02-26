import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import config


class BodyClothDataset(Dataset):

    def __init__(self, image_dir, body_csv):

        self.image_dir = image_dir
        self.body_df = pd.read_csv(body_csv)

        # Keep only rows that actually exist as images
        self.image_names = [
            img for img in self.body_df["image"].tolist()
            if os.path.exists(os.path.join(self.image_dir, img))
        ]

        self.body_df = self.body_df[
            self.body_df["image"].isin(self.image_names)
        ].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.body_df)

    def __getitem__(self, idx):

        row = self.body_df.iloc[idx]

        img_name = row["image"]
        img_path = os.path.join(self.image_dir, img_name)

        # Body vector
        body_vector = torch.tensor(
            row.drop("image").values,
            dtype=torch.float32
        )

        # Positive image
        pos_img = Image.open(img_path).convert("RGB")
        pos_img = self.transform(pos_img)

        # Negative image (random different index)
        neg_idx = random.randint(0, len(self.body_df) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.body_df) - 1)

        neg_img_name = self.body_df.iloc[neg_idx]["image"]
        neg_img_path = os.path.join(self.image_dir, neg_img_name)

        neg_img = Image.open(neg_img_path).convert("RGB")
        neg_img = self.transform(neg_img)

        return body_vector, pos_img, neg_img