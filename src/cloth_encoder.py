"""
cloth_encoder.py

  "We use a ResNet-50 pretrained on ImageNet to extract
   visual features from the catalog images, which captures
   the overall color, pattern, and silhouette of clothing."
   (2048-D CNN features)

  Full flow:
    CNN features (2048) → h_cnn (2-layer MLP) → reduced
    Attr features  (64) → h_attr(2-layer MLP) → reduced
    Concatenate → f_cloth (single FC) → 128-dim embedding

Our Simplification:
  We use CNN features only (no attribute text available
  in Fashionista dataset). 

Rules:
  ✓ Uses pretrained ResNet50 
  ✓ Classification head removed
  ✓ Projection layer → 128-dim
  ✓ Does NOT touch dataset
  ✓ Does NOT contain training logic
  ✓ Only produces embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# IMAGE PREPROCESSING

cloth_transform = transforms.Compose([
    transforms.Resize((224, 224)),        # ResNet standard input
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],       # ImageNet mean
        std =[0.229, 0.224, 0.225]        # ImageNet std
    )
])

# CLOTH ENCODER

class ClothEncoder(nn.Module):
    """
    Clothing Image Encoder 
    
    Maps clothing image → 128-dim embedding.

    Implements h_cnn + f_cloth :

        ResNet50   : pretrained CNN backbone (paper uses ResNet-50)
                     classification head REMOVED
                     outputs 2048-dim feature vector

        h_cnn      : 2-layer MLP (dimensionality reduction)
                     reduces 2048 → 256

        f_cloth    : single FC layer (projects to embedding space)
                     256 → 128

        L2 norm    : constrains to unit hypersphere (Section 3.3)

    Full Architecture:
        Input Image (3, 224, 224)
          ↓
        ┌──────────────────────────────────────┐
        │  ResNet50 Backbone (pretrained)       │
        │  [classification head REMOVED]        │
        │  Global Average Pool                  │
        │  Output: 2048-dim feature vector      │
        └──────────────────────────────────────┘
          ↓
        ┌──────────────────────────────────────┐
        │  h_cnn  (2-layer MLP)                │
        │  Linear(2048→512) + BN + ReLU        │
        │  Linear(512→256)  + BN + ReLU        │
        └──────────────────────────────────────┘
          ↓
        ┌──────────────────────────────────────┐
        │  f_cloth  (single FC layer)           │
        │  Linear(256→128)                      │
        └──────────────────────────────────────┘
          ↓
        L2 Normalize (unit hypersphere)
          ↓
        Output (128-dim continuous embedding)
    """

    def __init__(self, embedding_dim=128, freeze_backbone=True):
        super(ClothEncoder, self).__init__()

        self.embedding_dim = embedding_dim

        # ── ResNet50 Backbone  ──
        resnet      = models.resnet50(pretrained=True)
        backbone_dim = resnet.fc.in_features   

        # Remove classification head (fc layer)
        # Keep everything up to Global Average Pool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Output: (batch, 2048, 1, 1) → flattened to (batch, 2048)

        # Freeze early layers of backbone
        # Fine-tune only last 2 ResNet blocks
        if freeze_backbone:
            layers = list(self.backbone.children())
            for i, layer in enumerate(layers):
                if i < 7:   # freeze layers 0-6, train 7-8
                    for param in layer.parameters():
                        param.requires_grad = False

        # ── h_cnn: 2-layer MLP ───────────────────────────────
        
        self.h_cnn = nn.Sequential(

            # Layer 1: 2048 → 512
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            # Layer 2: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.f_cloth = nn.Linear(256, embedding_dim)

        # Initialize projection layers
        self._init_weights()

    def _init_weights(self):
        """Initialize only the new layers, not pretrained backbone."""
        for m in [self.h_cnn, self.f_cloth]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    elif isinstance(layer, nn.BatchNorm1d):
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x : Tensor (batch_size, 3, 224, 224) — clothing images

        Returns:
            embedding : Tensor (batch_size, 128)
                        L2-normalized, on unit hypersphere
                        CONTINUOUS — unique per clothing item
        """
        # Step 1: ResNet50 backbone — extract visual features
        features  = self.backbone(x)              # (batch, 2048, 1, 1)
        features  = features.view(features.size(0), -1)  # (batch, 2048)

        # Step 2: h_cnn — reduce to compact representation
        reduced   = self.h_cnn(features)          # (batch, 2048) → (batch, 256)

        # Step 3: f_cloth — project to embedding space
        embedding = self.f_cloth(reduced)         # (batch, 256) → (batch, 128)

        # Step 4: L2 normalize — unit hypersphere (paper Section 3.3)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding                          # (batch, 128)

    def get_embedding(self, image_path):
        """
        Convenience: image path → numpy embedding.

        Args:
            image_path : str, path to clothing image

        Returns:
            embedding  : numpy (128,)
        """
        image  = Image.open(image_path).convert("RGB")
        tensor = cloth_transform(image).unsqueeze(0)  # (1, 3, 224, 224)

        self.eval()
        with torch.no_grad():
            emb = self.forward(tensor)

        return emb.squeeze(0).numpy()  # (128,)

    def get_embedding_batch(self, image_paths):
        """
        Convenience: list of image paths → numpy embeddings.

        Args:
            image_paths : list of str

        Returns:
            embeddings  : numpy (N, 128)
        """
        tensors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensors.append(cloth_transform(img))

        batch = torch.stack(tensors)  # (N, 3, 224, 224)

        self.eval()
        with torch.no_grad():
            embs = self.forward(batch)

        return embs.numpy()  # (N, 128)

# QUICK TEST

if __name__ == "__main__":
    print("=" * 55)
    print("  ClothEncoder — ViBE Paper Architecture")
    print("=" * 55)

    model     = ClothEncoder(embedding_dim=128, freeze_backbone=True)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    print(f"\n  Backbone      : ResNet50 (pretrained ImageNet)")
    print(f"  Backbone out  : 2048-dim")
    print(f"  h_cnn         : 2-layer MLP  2048 → 512 → 256")
    print(f"  f_cloth       : single FC    256  → 128")
    print(f"  output_dim    : 128 (L2 normalized)")
    print(f"  Total params  : {total:,}")
    print(f"  Trainable     : {trainable:,}")

    # Test with batch of 4 clothing images
    dummy  = torch.randn(4, 3, 224, 224)
    output = model(dummy)

    print(f"\n  Input  shape  : {dummy.shape}")
    print(f"  Output shape  : {output.shape}")
    print(f"  L2 norms      : {output.norm(dim=1).tolist()}")
    print(f"  (All should be exactly 1.0)\n")
    print(f"  ✓ ClothEncoder ready — 128-dim continuous embeddings")
