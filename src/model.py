import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from body_encoder import BodyEncoder, BodyMeasurementExtractor
from cloth_encoder import ClothEncoder
from PIL import Image


class ViBEModel(nn.Module):

    def __init__(self,
                 body_input_dim=7,
                 embedding_dim=128):
        super().__init__()

        self.body_encoder = BodyEncoder(
            input_dim=body_input_dim,
            embedding_dim=embedding_dim
        )

        self.cloth_encoder = ClothEncoder(
            embedding_dim=embedding_dim,
            pretrained_path="/content/drive/MyDrive/SmartWardrobe/clothing_encoder.pth",
            freeze=False
        )

    def encode_body(self, body_features):
        return self.body_encoder(body_features)

    def encode_cloth(self, cloth_images):
        return self.cloth_encoder(cloth_images)

    def forward(self, body_features, cloth_images):
        body_emb = self.encode_body(body_features)
        cloth_emb = self.encode_cloth(cloth_images)
        return body_emb, cloth_emb

    def euclidean_distance(self, body_features, cloth_images):
        body_emb, cloth_emb = self.forward(body_features, cloth_images)
        return F.pairwise_distance(body_emb, cloth_emb, p=2)


# ================= RECOMMENDATION =================

def recommend_clothing_for_body(body_embedding, cloth_embeddings, cloth_names, top_k=5):

    body_t = torch.tensor(body_embedding, dtype=torch.float32).unsqueeze(0)
    cloth_t = torch.tensor(cloth_embeddings, dtype=torch.float32)

    distances = torch.cdist(body_t, cloth_t, p=2).squeeze(0).numpy()
    top_indices = np.argsort(distances)[:top_k]

    return [(cloth_names[i], round(float(distances[i]), 4)) for i in top_indices]


def recommend_body_for_clothing(cloth_embedding, body_embeddings, body_names, top_k=5):

    cloth_t = torch.tensor(cloth_embedding, dtype=torch.float32).unsqueeze(0)
    body_t = torch.tensor(body_embeddings, dtype=torch.float32)

    distances = torch.cdist(cloth_t, body_t, p=2).squeeze(0).numpy()
    top_indices = np.argsort(distances)[:top_k]

    return [(body_names[i], round(float(distances[i]), 4)) for i in top_indices]


# ================= SAVE / LOAD =================

def save_model(model, path="vibe_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"[✓] Model saved → {path}")


def load_model(path="vibe_model.pth",
               body_input_dim=7,
               embedding_dim=128,
               device='cpu'):

    model = ViBEModel(body_input_dim, embedding_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    print(f"[✓] Model loaded ← {path}")
    return model


# ================= TEST =================

if __name__ == "__main__":

    print("=" * 55)
    print("  ViBEModel — Joint Embedding Test")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = ViBEModel().to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal params  : {total:,}")
    print(f"Trainable     : {trainable:,}")

    # Dummy input
    dummy_body = torch.randn(4, 7).to(device)
    dummy_cloth = torch.randn(4, 3, 224, 224).to(device)

    body_emb, cloth_emb = model(dummy_body, dummy_cloth)

    print(f"\nBody emb shape  : {body_emb.shape}")
    print(f"Cloth emb shape : {cloth_emb.shape}")
    print(f"Body norms      : {body_emb.norm(dim=1).tolist()}")
    print(f"Cloth norms     : {cloth_emb.norm(dim=1).tolist()}")

    distances = model.euclidean_distance(dummy_body, dummy_cloth)

    print(f"\nDistances: {distances.tolist()}")

    print("\n✓ Model working correctly")