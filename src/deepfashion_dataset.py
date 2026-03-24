import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepFashionDataset(Dataset):
    def __init__(self, root_dir):

        self.root_dir = root_dir

        # Load annotations
        with open(os.path.join(root_dir, "annotations.json"), "r") as f:
            self.data = json.load(f)

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
        return len(self.data)

    def __getitem__(self, idx):

        img_path, label = self.data[idx]

        full_path = os.path.join(self.root_dir, img_path)

        img = Image.open(full_path).convert("RGB")
        img = self.transform(img)

        return img, label