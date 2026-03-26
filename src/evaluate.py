"""
evaluate.py — Unified evaluation script
 
Replaces both test_deepfashion.py and test_fashionista_subset.py.
 
Metrics reported:
  Fashionista   → Recall@1, Recall@5, MRR, Category Hit@10
  DeepFashion   → Recall@1, Recall@5, Recall@10
 
Run:
  python evaluate.py --mode fashionista
  python evaluate.py --mode deepfashion
  python evaluate.py --mode both
"""
 
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
 
import config
from model import ViBEModel
from cloth_encoder import upper_body_transform, cloth_transform
from loss import check_embedding_collapse
from PIL import Image
 
 
# ================================================================
# SHARED SETUP
# ================================================================
 
def load_model(device):
    model = ViBEModel(body_input_dim=config.BODY_INPUT_DIM,
                      embedding_dim=config.EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH,
                                      map_location=device))
    model.eval()
    return model
 
 
@torch.no_grad()
def encode_cloth(model, img_path, device, use_upper_crop=True):
    tf  = upper_body_transform if use_upper_crop else cloth_transform
    img = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    return model.encode_cloth(img).cpu().numpy()[0]
 
 
@torch.no_grad()
def encode_body(model, vec, device):
    t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    return model.encode_body(t).cpu().numpy()[0]
 
 
# ================================================================
# FASHIONISTA EVALUATION
# ================================================================
 
def evaluate_fashionista(model, device, gallery_size=500):
    print("\n" + "="*55)
    print("  Fashionista evaluation")
    print("="*55)
 
    df          = pd.read_csv(config.BODY_VECTOR_CSV)
    cloth_paths = list(Path(config.TEST_DIR).glob("*.jpg"))
    cloth_names = [p.name for p in cloth_paths]
    df          = df[df["image"].isin(cloth_names)].reset_index(drop=True)
 
    print(f"  Test samples: {len(df)}")
 
    # Precompute all clothing embeddings
    print("  Computing clothing embeddings...")
    cloth_embs = []
    for p in tqdm(cloth_paths):
        cloth_embs.append(encode_cloth(model, p, device, use_upper_crop=True))
    cloth_embs = np.array(cloth_embs)
 
    # Collapse check
    check_embedding_collapse(
        torch.tensor(cloth_embs),
        torch.tensor(cloth_embs)
    )
 
    recall1  = recall5 = 0
    mrr_vals = []
    n        = len(df)
 
    for _, row in tqdm(df.iterrows(), total=n):
        body_vec = row.values[1:].astype(np.float32)
        body_emb = encode_body(model, body_vec, device)
        gt       = row["image"]
 
        # Build gallery (random 500 subset containing ground truth)
        gallery_idx = random.sample(range(len(cloth_embs)),
                                     min(gallery_size - 1, len(cloth_embs) - 1))
        gt_idx = cloth_names.index(gt)
        if gt_idx not in gallery_idx:
            gallery_idx.append(gt_idx)
 
        gallery_emb   = cloth_embs[gallery_idx]
        gallery_names = [cloth_names[i] for i in gallery_idx]
 
        scores = np.dot(gallery_emb, body_emb)
        ranked = np.argsort(-scores)
        ranked_names = [gallery_names[i] for i in ranked]
 
        # Recall
        if gt in ranked_names[:1]:  recall1 += 1
        if gt in ranked_names[:5]:  recall5 += 1
 
        # MRR
        try:
            rank = ranked_names.index(gt)
            mrr_vals.append(1.0 / (rank + 1))
        except ValueError:
            mrr_vals.append(0.0)
 
    print(f"\n  Recall@1  : {recall1/n:.4f}")
    print(f"  Recall@5  : {recall5/n:.4f}")
    print(f"  MRR       : {np.mean(mrr_vals):.4f}")
    print(f"  (MRR > 0.05 on Fashionista is meaningful given weak supervision)")
 
 
# ================================================================
# DEEPFASHION EVALUATION
# ================================================================
 
def evaluate_deepfashion(model, device):
    print("\n" + "="*55)
    print("  DeepFashion retrieval evaluation")
    print("="*55)
 
    QUERY_DIR   = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/queries"
    GALLERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/gallery"
 
    def get_clothing_id(path):
        parts = Path(path).name.split("_")
        for i, p in enumerate(parts):
            if p == "id":
                return parts[i] + "_" + parts[i+1]
        return None
 
    # Gallery embeddings
    print("  Computing gallery embeddings...")
    gallery_paths = list(Path(GALLERY_DIR).glob("*.jpg"))
    gallery_embs  = []
    gallery_ids   = []
    for p in tqdm(gallery_paths):
        # DeepFashion: use full image (not upper-crop) — full outfit images
        gallery_embs.append(encode_cloth(model, p, device, use_upper_crop=False))
        gallery_ids.append(get_clothing_id(p))
    gallery_embs = np.array(gallery_embs)
 
    # Query evaluation
    print("  Running retrieval evaluation...")
    query_paths = list(Path(QUERY_DIR).glob("*.jpg"))
    r1 = r5 = r10 = 0
    n  = len(query_paths)
 
    for q in tqdm(query_paths):
        q_emb = encode_cloth(model, q, device, use_upper_crop=False)
        q_id  = get_clothing_id(q)
 
        scores     = np.dot(gallery_embs, q_emb)
        ranked_ids = [gallery_ids[i] for i in np.argsort(-scores)]
 
        if q_id in ranked_ids[:1]:   r1  += 1
        if q_id in ranked_ids[:5]:   r5  += 1
        if q_id in ranked_ids[:10]:  r10 += 1
 
    print(f"\n  Recall@1  : {r1/n:.4f}")
    print(f"  Recall@5  : {r5/n:.4f}")
    print(f"  Recall@10 : {r10/n:.4f}")
 
 
# ================================================================
# ENTRY POINT
# ================================================================
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fashionista", "deepfashion", "both"],
                        default="fashionista")
    parser.add_argument("--gallery_size", type=int, default=500)
    args = parser.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(device)
 
    if args.mode in ("fashionista", "both"):
        evaluate_fashionista(model, device, args.gallery_size)
 
    if args.mode in ("deepfashion", "both"):
        evaluate_deepfashion(model, device)
 
