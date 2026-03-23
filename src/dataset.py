import os
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

        # Keep only valid images
        self.image_names = [
            img for img in self.body_df["image"].tolist()
            if os.path.exists(os.path.join(self.image_dir, img))
        ]

        self.body_df = self.body_df[
            self.body_df["image"].isin(self.image_names)
        ].reset_index(drop=True)

        # ✅ Improved transform (VERY IMPORTANT)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.body_df)

    def __getitem__(self, idx):

        row = self.body_df.iloc[idx]

        img_name = row["image"]
        img_path = os.path.join(self.image_dir, img_name)

        # Body vector
        body_vector = torch.tensor(
            row.drop("image").to_numpy(dtype=float),
            dtype=torch.float32
        )

        # Clothing image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # ✅ Return ONLY (no negative image now)
        return {
            "image": img,
            "body": body_vector
        }