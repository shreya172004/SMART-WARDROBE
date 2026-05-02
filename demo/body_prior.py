
"""
body_prior.py — Body-Shape Compatibility Prior
 
PURPOSE
-------
After cosine similarity ranking, some retrieved items may be
statistically uncommon for the user's body shape. This module
builds a lightweight prior from your existing Fashionista training
data and uses it to re-rank recommendations at inference time.
 
HOW IT WORKS
------------
1. Load training body vectors + clothing embeddings
2. Cluster body vectors into K shape groups (e.g. 6)
3. For each cluster, compute the centroid of its clothing embeddings
4. At inference: find user's nearest body cluster, compute similarity
   between candidate items and that cluster's centroid, blend with
   the main cosine score
 
This requires NO retraining — pure post-processing on existing data.
It takes ~3 minutes to build the prior (run once, save to disk).
 
WHAT IT DOES NOT DO
-------------------
- It does not label items as "flattering" or "unflattering"
- It does not use hand-coded fashion rules
- It does not require any new dataset
- It does not change the model weights
 
It learns from data: "users with body shape X historically wore
clothing items that cluster around embedding region Y."
 
USAGE
-----
    # Build prior once (saves to disk)
    python body_prior.py --build \
        --train_dir /path/to/split/train \
        --body_csv  /path/to/body_vectors.csv \
        --model_path /path/to/best_vibe_model.pth \
        --save_path /path/to/body_prior.pkl
 
    # Use at inference (in recommender.py)
    from body_prior import BodyShapePrior
    prior = BodyShapePrior.load("body_prior.pkl")
    final_scores = prior.rerank(body_vec, cloth_embs, base_scores, alpha=0.3)
"""
 
import os
import pickle
import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
 
sys.path.insert(0, "/content/SMART-WARDROBE/src")
import config
from model import ViBEModel
from dataset import BodyClothDataset
 
 
# ================================================================
# PRIOR CLASS
# ================================================================
 
class BodyShapePrior:
    """
    Body-shape compatibility prior.
 
    Attributes:
        n_clusters     : number of body shape groups
        scaler         : StandardScaler fitted on training body vectors
        kmeans         : KMeans fitted on scaled body vectors
        cloth_centroids: (n_clusters, 128) — mean cloth embedding per cluster
        cluster_counts  : (n_clusters,)    — number of samples per cluster
    """
 
    def __init__(self, n_clusters: int = 6):
        self.n_clusters      = n_clusters
        self.scaler          = StandardScaler()
        self.kmeans          = KMeans(n_clusters=n_clusters, random_state=42,
                                      n_init=10)
        self.cloth_centroids = None   # filled by .build()
        self.cluster_counts  = None
 
    # ── Build ────────────────────────────────────────────────────────
 
    def build(self, body_vecs: np.ndarray, cloth_embs: np.ndarray):
        """
        Fit the prior from training data.
 
        body_vecs  : (N, 7)   raw body measurement vectors
        cloth_embs : (N, 128) L2-normalised clothing embeddings
        """
        assert len(body_vecs) == len(cloth_embs), \
            f"Mismatch: {len(body_vecs)} body vecs vs {len(cloth_embs)} cloth embs"
 
        print(f"  Building prior from {len(body_vecs)} training samples...")
 
        # Step 1: standardise body vectors
        body_scaled = self.scaler.fit_transform(body_vecs)
 
        # Step 2: cluster into K body shape groups
        self.kmeans.fit(body_scaled)
        labels = self.kmeans.labels_
        print(f"  Cluster distribution: "
              f"{np.bincount(labels).tolist()}")
 
        # Step 3: for each cluster, compute mean clothing embedding
        self.cloth_centroids = np.zeros((self.n_clusters, cloth_embs.shape[1]),
                                         dtype=np.float32)
        self.cluster_counts  = np.zeros(self.n_clusters, dtype=np.int32)
 
        for k in range(self.n_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                centroid = cloth_embs[mask].mean(axis=0)
                # L2 normalise the centroid
                norm = np.linalg.norm(centroid)
                self.cloth_centroids[k] = centroid / (norm + 1e-8)
                self.cluster_counts[k]  = mask.sum()
 
        print(f"  Cloth centroids computed for {self.n_clusters} clusters.")
        print(f"  Prior ready.")
 
    # ── Assign cluster ───────────────────────────────────────────────
 
    def get_cluster(self, body_vec: np.ndarray) -> int:
        """Return the cluster index for a single body vector."""
        scaled = self.scaler.transform(body_vec.reshape(1, -1))
        return int(self.kmeans.predict(scaled)[0])
 
    # ── Re-rank ──────────────────────────────────────────────────────
 
    def rerank(
        self,
        body_vec:    np.ndarray,
        cloth_embs:  np.ndarray,
        base_scores: np.ndarray,
        alpha:       float = 0.25,
    ) -> np.ndarray:
        """
        Blend model cosine scores with body-shape prior.
 
        base_scores : (N,) cosine similarities from the model
        cloth_embs  : (N, 128) L2-normalised clothing embeddings
        body_vec    : (7,)  raw body vector of the current user
        alpha       : weight of prior (0 = pure model, 1 = pure prior)
 
        Returns:
            (N,) blended scores — use these for final ranking
        """
        cluster_id = self.get_cluster(body_vec)
        centroid   = self.cloth_centroids[cluster_id]          # (128,)
 
        # Prior score: cosine similarity between each item and the
        # cluster centroid (= typical clothing for this body shape)
        prior_scores = cloth_embs @ centroid                   # (N,)
 
        # Normalise both score arrays to [0, 1] for fair blending
        def _norm01(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-8)
 
        base_n  = _norm01(base_scores)
        prior_n = _norm01(prior_scores)
 
        blended = (1 - alpha) * base_n + alpha * prior_n
        return blended
 
    # ── Save / Load ──────────────────────────────────────────────────
 
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  Prior saved → {path}")
 
    @classmethod
    def load(cls, path: str) -> "BodyShapePrior":
        with open(path, "rb") as f:
            prior = pickle.load(f)
        print(f"  Prior loaded ← {path}")
        return prior
 
 
# ================================================================
# BUILD SCRIPT
# ================================================================
 
def build_prior(
    train_dir:  str,
    body_csv:   str,
    model_path: str,
    save_path:  str,
    n_clusters: int = 6,
    batch_size: int = 64,
):
    """
    Build and save the body-shape prior from training data.
    Run this once — takes ~3–5 minutes depending on dataset size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBuilding body-shape prior on {device}")
 
    # Load model
    model = ViBEModel(
        body_input_dim=config.BODY_INPUT_DIM,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Model loaded from {model_path}")
 
    # Load training dataset
    dataset = BodyClothDataset(
        image_dir=train_dir,
        body_csv=body_csv,
        augment=False,
        body_completeness_threshold=0.7
    )
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Dataset: {len(dataset)} samples")
 
    # Encode all training samples
    all_body_vecs  = []
    all_cloth_embs = []
 
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding training set"):
            body_vec  = batch["body"].to(device)
            cloth_img = batch["image"].to(device)
 
            cloth_emb = F.normalize(model.encode_cloth(cloth_img), dim=1)
 
            all_body_vecs.append(batch["body"].numpy())   # raw, not encoded
            all_cloth_embs.append(cloth_emb.cpu().numpy())
 
    body_vecs  = np.concatenate(all_body_vecs,  axis=0)
    cloth_embs = np.concatenate(all_cloth_embs, axis=0)
 
    print(f"  Encoded: {len(body_vecs)} body vectors, "
          f"{len(cloth_embs)} cloth embeddings")
 
    # Build and save prior
    prior = BodyShapePrior(n_clusters=n_clusters)
    prior.build(body_vecs, cloth_embs)
    prior.save(save_path)
 
    return prior
 
 
# ================================================================
# ENTRY POINT
# ================================================================
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build body-shape compatibility prior"
    )
    parser.add_argument("--build",      action="store_true")
    parser.add_argument("--train_dir",  default=config.TRAIN_DIR)
    parser.add_argument("--body_csv",   default=config.BODY_VECTOR_CSV)
    parser.add_argument("--model_path", default=config.BEST_MODEL_PATH)
    parser.add_argument("--save_path",
                        default="/content/drive/MyDrive/SmartWardrobe/body_prior.pkl")
    parser.add_argument("--n_clusters", type=int, default=6)
    args = parser.parse_args()
 
    if args.build:
        build_prior(
            train_dir=args.train_dir,
            body_csv=args.body_csv,
            model_path=args.model_path,
            save_path=args.save_path,
            n_clusters=args.n_clusters,
        )
    else:
        parser.print_help()
