import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
 
 
# ================================================================
# TRANSFORMS
# ================================================================
 
# Full-image transform (used during Fashionista training)
cloth_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])
 
# Upper-body crop transform — reduces pose/shoe/background bias.
# KEY FIX: this is why the model was recommending heels — it matched
# full-image visual similarity (dark tones, similar framing) rather
# than garment compatibility. Cropping to the torso region forces the
# encoder to focus on actual clothing.
class UpperBodyCropTransform:
    """Crop top 65% of image (torso region) then resize to 224."""
    def __init__(self):
        self.base = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])
        self.resize = transforms.Resize((224, 224))
 
    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        img  = img.crop((0, 0, w, int(h * 0.65)))   # drop bottom 35%
        img  = self.resize(img)
        return self.base(img)
 
upper_body_transform = UpperBodyCropTransform()
 
 
# ================================================================
# ENCODER
# ================================================================
 
class ClothEncoder(nn.Module):
    """
    ResNet-50 backbone + 2-layer projection head → 128-D L2 embedding.
 
    Changes from v1:
    - unfreeze_layer4_only() replaces full unfreeze_backbone()
      → prevents the pretrained visual features from being overwritten
      by Fashionista's noisy supervision
    - projector now has BatchNorm (matches pretraining architecture)
    """
 
    def __init__(self, embedding_dim=128, pretrained_path=None, freeze=False):
        super().__init__()
 
        # Backbone
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone_dim   = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
 
        # Projector — must match pretrain_clothing_encoder.py architecture
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
 
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            self.load_state_dict(state, strict=False)
            print(f"  Loaded pretrained weights from {pretrained_path}")
 
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
 
    def forward(self, x):
        features  = self.backbone(x)
        embedding = self.projector(features)
        embedding = F.normalize(embedding, dim=1)
        return embedding
 
    # ── Freeze helpers ──────────────────────────────────────────────
 
    def freeze_backbone(self):
        """Freeze entire backbone (stage 1 of fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
 
    def unfreeze_layer4_only(self):
        """
        Unfreeze ONLY the last residual block (layer4) + projector.
        REPLACES the old unfreeze_backbone() which unfroze everything
        and caused overfitting after epoch 4.
        """
        # Keep everything frozen first
        for param in self.backbone.parameters():
            param.requires_grad = False
 
        # Unfreeze only layer4 (the last residual block)
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
 
        # Projector always trainable
        for param in self.projector.parameters():
            param.requires_grad = True
 
    def unfreeze_backbone(self):
        """Full unfreeze — only use this after Polyvore pretraining."""
        for param in self.backbone.parameters():
            param.requires_grad = True
 
    # ── Inference ───────────────────────────────────────────────────
 
    def get_embedding(self, image_path, use_upper_crop=True):
        tf  = upper_body_transform if use_upper_crop else cloth_transform
        img = Image.open(image_path).convert("RGB")
        t   = tf(img).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            emb = self.forward(t)
        return emb.squeeze(0).cpu().numpy()
 
    def get_embedding_batch(self, image_paths, use_upper_crop=True):
        tf      = upper_body_transform if use_upper_crop else cloth_transform
        tensors = [tf(Image.open(p).convert("RGB")) for p in image_paths]
        batch   = torch.stack(tensors)
        self.eval()
        with torch.no_grad():
            embs = self.forward(batch)
        return embs.cpu().numpy()
 
