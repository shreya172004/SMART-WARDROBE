import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image


# ================= TRANSFORM =================
cloth_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


class ClothEncoder(nn.Module):

    def __init__(self, embedding_dim=128, pretrained_path=None, freeze=False):
        super().__init__()

        # ================= BACKBONE =================
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # ================= PROJECTOR (MATCH PRETRAINING) =================
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, embedding_dim)
        )

        # ================= LOAD FULL PRETRAINED =================
        if pretrained_path:
            self.load_state_dict(
                torch.load(pretrained_path, map_location="cpu"),
                strict=False
            )

        # ================= FREEZE OPTION =================
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):

        features = self.backbone(x)
        embedding = self.projector(features)

        # 🔥 CRITICAL
        embedding = F.normalize(embedding, dim=1)

        return embedding


    # ================= INFERENCE UTILITIES =================

    def get_embedding(self, image_path):

        image = Image.open(image_path).convert("RGB")
        tensor = cloth_transform(image).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            emb = self.forward(tensor)

        return emb.squeeze(0).cpu().numpy()


    def get_embedding_batch(self, image_paths):

        tensors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensors.append(cloth_transform(img))

        batch = torch.stack(tensors)

        self.eval()
        with torch.no_grad():
            embs = self.forward(batch)

        return embs.cpu().numpy()