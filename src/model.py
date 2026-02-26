"""
model.py
═══════════════════════════════════════════════════════════════
Implements the JOINT EMBEDDING MODEL from the ViBE paper:
  "ViBE: Dressing for Diverse Body Shapes" (CVPR 2020)

  " We have two kinds of triplets,
   one between bodies and clothing, and one between bodies
   and bodies. So our final loss combines two instances of
   the margin-based loss:
       L = L_body,cloth + L_body,body"

  "We constrain the embedding to live on the d-dimensional
   hypersphere for training stability."

  Distance metric: Euclidean distance D(.,.)
  NOT cosine similarity, NOT bilinear similarity.

This file:
  - Combines BodyEncoder + ClothEncoder
  - Both map into the SAME 128-dim shared space
  - Provides similarity and recommendation functions
  - Does NOT contain dataset logic
  - Does NOT contain training loops

Rules:
  ✓ Combines body_encoder + cloth_encoder
  ✓ Shared 128-dim embedding space
  ✓ Euclidean distance (as in paper)
  ✓ Does NOT touch dataset
  ✓ Does NOT contain training logic
  ✓ Only produces embeddings + similarity scores
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from body_encoder import BodyEncoder, BodyMeasurementExtractor
from cloth_encoder import ClothEncoder, cloth_transform
from PIL import Image


# ═══════════════════════════════════════════════════════════════
# VIBE JOINT EMBEDDING MODEL
# ═══════════════════════════════════════════════════════════════

class ViBEModel(nn.Module):

    def __init__(self,
                 body_input_dim  = 7,
                 embedding_dim   = 128,
                 freeze_backbone = True):
        super(ViBEModel, self).__init__()

        # ── Body Encoder ──────────────────────────────────────
        # input_dim=7 (measurements), output=128-dim
        self.body_encoder  = BodyEncoder(
            input_dim    = body_input_dim,
            embedding_dim= embedding_dim
        )

        # ── Cloth Encoder ─────────────────────────────────────
        # ResNet50 backbone, output=128-dim
        self.cloth_encoder = ClothEncoder(
            embedding_dim   = embedding_dim,
            freeze_backbone = freeze_backbone
        )

        self.embedding_dim = embedding_dim

    def encode_body(self, body_features):
        """
        Encode body measurements → 128-dim embedding.

        Args:
            body_features : Tensor (batch, 7)

        Returns:
            embedding     : Tensor (batch, 128), L2-normalized
        """
        return self.body_encoder(body_features)

    def encode_cloth(self, cloth_images):
        """
        Encode clothing images → 128-dim embedding.

        Args:
            cloth_images : Tensor (batch, 3, 224, 224)

        Returns:
            embedding    : Tensor (batch, 128), L2-normalized
        """
        return self.cloth_encoder(cloth_images)

    def forward(self, body_features, cloth_images):
        """
        Forward pass — returns both embeddings.

        Args:
            body_features : Tensor (batch, 7)
            cloth_images  : Tensor (batch, 3, 224, 224)

        Returns:
            body_emb  : Tensor (batch, 128)
            cloth_emb : Tensor (batch, 128)
        """
        body_emb  = self.encode_body(body_features)
        cloth_emb = self.encode_cloth(cloth_images)
        return body_emb, cloth_emb

    def euclidean_distance(self, body_features, cloth_images):
        """
        Compute Euclidean distance between body and cloth embeddings.

        Paper uses Euclidean distance D(.,.) — NOT cosine similarity.

        Args:
            body_features : Tensor (batch, 7)
            cloth_images  : Tensor (batch, 3, 224, 224)

        Returns:
            distances : Tensor (batch,)
                        Lower = more compatible
        """
        body_emb, cloth_emb = self.forward(body_features, cloth_images)
        return F.pairwise_distance(body_emb, cloth_emb, p=2)

# Using Euclidean distance 

def recommend_clothing_for_body(body_embedding,
                                 cloth_embeddings,
                                 cloth_names,
                                 top_k=5):
    """
    Given a body embedding, find top-k most compatible clothing.

    Uses Euclidean distance — smaller = more compatible.

    Args:
        body_embedding   : numpy (128,)
        cloth_embeddings : numpy (N, 128)
        cloth_names      : list of N item names/paths
        top_k            : int

    Returns:
        List of (name, distance) sorted by ascending distance
    """
    body_t  = torch.tensor(body_embedding,   dtype=torch.float32).unsqueeze(0)
    cloth_t = torch.tensor(cloth_embeddings, dtype=torch.float32)

    # Euclidean distances
    distances   = torch.cdist(body_t, cloth_t, p=2).squeeze(0).numpy()

    # Sort ascending (closest first)
    top_indices = np.argsort(distances)[:top_k]

    return [(cloth_names[i], round(float(distances[i]), 4))
            for i in top_indices]


def recommend_body_for_clothing(cloth_embedding,
                                 body_embeddings,
                                 body_names,
                                 top_k=5):
    """
    Vice versa direction — given clothing, find most compatible bodies.

    Args:
        cloth_embedding  : numpy (128,)
        body_embeddings  : numpy (N, 128)
        body_names       : list of N body names/paths
        top_k            : int

    Returns:
        List of (name, distance) sorted by ascending distance
    """
    cloth_t = torch.tensor(cloth_embedding,  dtype=torch.float32).unsqueeze(0)
    body_t  = torch.tensor(body_embeddings,  dtype=torch.float32)

    distances   = torch.cdist(cloth_t, body_t, p=2).squeeze(0).numpy()
    top_indices = np.argsort(distances)[:top_k]

    return [(body_names[i], round(float(distances[i]), 4))
            for i in top_indices]

def save_model(model, path="vibe_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"[✓] Model saved → {path}")


def load_model(path="vibe_model.pth",
               body_input_dim=7,
               embedding_dim=128,
               device='cpu'):
    model = ViBEModel(body_input_dim=body_input_dim,
                      embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[✓] Model loaded ← {path}")
    return model

#TEST

if __name__ == "__main__":
    print("=" * 55)
    print("  ViBEModel — Joint Embedding Test")
    print("=" * 55)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    model     = ViBEModel(body_input_dim=7, embedding_dim=128)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    print(f"\n  BodyEncoder   : input=7 → h_meas(MLP) → f_body → 128")
    print(f"  ClothEncoder  : ResNet50 → h_cnn(MLP) → f_cloth → 128")
    print(f"  Shared space  : 128-dim unit hypersphere")
    print(f"  Distance      : Euclidean (as in paper)")
    print(f"  Total params  : {total:,}")
    print(f"  Trainable     : {trainable:,}")

    # ── Test Forward Pass ──────────────────────────────────────
    dummy_body  = torch.randn(4, 7)              # 4 people
    dummy_cloth = torch.randn(4, 3, 224, 224)    # 4 clothing items

    body_emb, cloth_emb = model(dummy_body, dummy_cloth)

    print(f"\n  ── Forward Pass ──")
    print(f"  Body  input  : {dummy_body.shape}")
    print(f"  Cloth input  : {dummy_cloth.shape}")
    print(f"  Body  emb    : {body_emb.shape}")
    print(f"  Cloth emb    : {cloth_emb.shape}")
    print(f"  Body  norms  : {body_emb.norm(dim=1).tolist()}")
    print(f"  Cloth norms  : {cloth_emb.norm(dim=1).tolist()}")

    # ── Test Euclidean Distance ────────────────────────────────
    distances = model.euclidean_distance(dummy_body, dummy_cloth)
    print(f"\n  ── Euclidean Distances (body ↔ cloth) ──")
    print(f"  {distances.tolist()}")
    print(f"  (Lower = more compatible)")

    # ── Test Recommendation ────────────────────────────────────
    body_np  = body_emb[0].detach().numpy()
    cloth_np = cloth_emb.detach().numpy()
    names    = ["item_A", "item_B", "item_C", "item_D"]

    print(f"\n  ── Recommendation Test (body → clothing) ──")
    recs = recommend_clothing_for_body(body_np, cloth_np, names, top_k=3)
    for name, dist in recs:
        print(f"  → {name}   distance: {dist}")

    print(f"\n  ── Recommendation Test (clothing → body) ──")
    recs2 = recommend_body_for_clothing(cloth_np[0], body_np[np.newaxis,:].repeat(4,0), names, top_k=3)
    for name, dist in recs2:
        print(f"  → {name}   distance: {dist}")

    print(f"\n  ✓ ViBEModel working")
    print(f"  ✓ Both encoders output 128-dim embeddings")
    print(f"  ✓ Shared embedding space with Euclidean distance")
    print(f"  ✓ Bidirectional recommendation ready")
