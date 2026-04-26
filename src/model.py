# ================================================================
# MODEL — ViBE (Visual + Body Embedding)
# ================================================================
# This model maps:
# ✔ Body features → embedding
# ✔ Clothing images → embedding
#
# Both embeddings lie in the SAME space → cosine similarity used
# for recommendation.
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import config


# ================================================================
# CLOTHING ENCODER (ResNet50 + projection head)
# ================================================================
class ClothingEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # Pretrained ResNet50 backbone
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()   # remove classification head
        self.backbone = backbone

        # Projection head → maps features to embedding space
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.projector(feat)

        # L2 normalize → cosine similarity space
        emb = F.normalize(emb, dim=1)
        return emb

    # Freeze entire backbone (used in early training)
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True

    # Fine-tune only layer4 (best practice)
    def unfreeze_layer4_only(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True
        for p in self.projector.parameters():
            p.requires_grad = True


# ================================================================
# BODY ENCODER (Improved MLP with BatchNorm)
# ================================================================
class BodyEncoder(nn.Module):
    def __init__(self, input_dim=7, embedding_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),   # stabilizes training
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        emb = self.net(x)

        # Normalize in embedding space (IMPORTANT)
        emb = F.normalize(emb, dim=1)
        return emb


# ================================================================
# MAIN MODEL
# ================================================================
class ViBEModel(nn.Module):
    """
    Visual-Body Embedding Model

    Learns a shared embedding space where:
    ✔ Compatible (body, clothing) pairs are close
    ✔ Incompatible pairs are far apart
    """

    def __init__(self,
                 body_input_dim=config.BODY_INPUT_DIM,
                 embedding_dim=config.EMBEDDING_DIM):
        super().__init__()

        self.body_encoder = BodyEncoder(body_input_dim, embedding_dim)
        self.cloth_encoder = ClothingEncoder(embedding_dim)

    def encode_body(self, body_vec):
        return self.body_encoder(body_vec)

    def encode_cloth(self, cloth_img):
        return self.cloth_encoder(cloth_img)

    def forward(self, body_vec, cloth_img):
        body_emb = self.encode_body(body_vec)
        cloth_emb = self.encode_cloth(cloth_img)
        return body_emb, cloth_emb


# ================================================================
# RECOMMENDATION UTILITY
# ================================================================
def recommend_clothing_for_body(body_embedding,
                               cloth_embeddings,
                               cloth_names,
                               top_k=5):
    """
    Rank clothing items using cosine similarity.
    """

    body_t  = torch.tensor(body_embedding, dtype=torch.float32).unsqueeze(0)
    cloth_t = torch.tensor(cloth_embeddings, dtype=torch.float32)

    # cosine similarity (dot product since normalized)
    scores = torch.matmul(cloth_t, body_t.T).squeeze(1).numpy()

    top_idx = np.argsort(-scores)[:top_k]

    return [(cloth_names[i], round(float(scores[i]), 4)) for i in top_idx]


# ================================================================
# SAVE / LOAD UTILITIES
# ================================================================
def save_model(model, path=config.BEST_MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"[✓] Model saved → {path}")


def load_model(path=config.BEST_MODEL_PATH,
               body_input_dim=config.BODY_INPUT_DIM,
               embedding_dim=config.EMBEDDING_DIM,
               device="cpu"):

    model = ViBEModel(body_input_dim, embedding_dim)
    model.load_state_dict(torch.load(path, map_location=device))

    model.to(device)
    model.eval()

    print(f"[✓] Model loaded ← {path}")
    return model


# ================================================================
# DEBUG / TEST RUN
# ================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViBEModel().to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")

    # Dummy input
    dummy_body  = torch.randn(4, 7).to(device)
    dummy_cloth = torch.randn(4, 3, 224, 224).to(device)

    body_emb, cloth_emb = model(dummy_body, dummy_cloth)

    print(f"Body emb shape   : {body_emb.shape}")
    print(f"Cloth emb shape  : {cloth_emb.shape}")
    print(f"Body norms       : {body_emb.norm(dim=1).tolist()}")
    print(f"Cloth norms      : {cloth_emb.norm(dim=1).tolist()}")