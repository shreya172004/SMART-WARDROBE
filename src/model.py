import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
from body_encoder import BodyEncoder
from cloth_encoder import ClothEncoder
 
 
class ViBEModel(nn.Module):
    """
    Visual-Body Embedding model.
 
    Maps body measurements and clothing images into a shared 128-D
    L2-normalized embedding space where compatible pairs are close.
 
    No architecture changes from v1 — the encoders themselves are
    the same. What changed is the training pipeline (see train.py).
    """
 
    def __init__(self, body_input_dim: int = 7, embedding_dim: int = 128):
        super().__init__()
 
        self.body_encoder  = BodyEncoder(
            input_dim=body_input_dim,
            embedding_dim=embedding_dim
        )
        self.cloth_encoder = ClothEncoder(
            embedding_dim=embedding_dim,
            freeze=False      # freeze is controlled externally in train.py
        )
 
    def encode_body(self, body_features: torch.Tensor) -> torch.Tensor:
        return self.body_encoder(body_features)
 
    def encode_cloth(self, cloth_images: torch.Tensor) -> torch.Tensor:
        return self.cloth_encoder(cloth_images)
 
    def forward(self, body_features, cloth_images):
        body_emb  = self.encode_body(body_features)
        cloth_emb = self.encode_cloth(cloth_images)
        return body_emb, cloth_emb
 
 
# ================================================================
# RECOMMENDATION UTILITIES
# ================================================================
 
def recommend_clothing_for_body(body_embedding, cloth_embeddings,
                                 cloth_names, top_k=5):
    """Cosine similarity ranking (use this — not Euclidean distance)."""
    body_t  = torch.tensor(body_embedding, dtype=torch.float32).unsqueeze(0)
    cloth_t = torch.tensor(cloth_embeddings, dtype=torch.float32)
 
    # Both are L2-normalized so cosine = dot product
    scores = torch.matmul(cloth_t, body_t.T).squeeze(1).numpy()
    top_idx = np.argsort(-scores)[:top_k]
    return [(cloth_names[i], round(float(scores[i]), 4)) for i in top_idx]
 
 
# ================================================================
# SAVE / LOAD
# ================================================================
 
def save_model(model, path="vibe_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"[✓] Model saved → {path}")
 
 
def load_model(path="vibe_model.pth", body_input_dim=7,
               embedding_dim=128, device="cpu"):
    model = ViBEModel(body_input_dim, embedding_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[✓] Model loaded ← {path}")
    return model
 
 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ViBEModel().to(device)
    model.eval()
 
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
 
    dummy_body  = torch.randn(4, 7).to(device)
    dummy_cloth = torch.randn(4, 3, 224, 224).to(device)
    body_emb, cloth_emb = model(dummy_body, dummy_cloth)
 
    print(f"Body emb shape   : {body_emb.shape}")
    print(f"Cloth emb shape  : {cloth_emb.shape}")
    print(f"Body norms       : {body_emb.norm(dim=1).tolist()}")
    print(f"Cloth norms      : {cloth_emb.norm(dim=1).tolist()}")
 

