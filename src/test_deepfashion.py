# ================================================================
# DEEPFASHION RETRIEVAL EVALUATION
# ================================================================

import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

import config
from model import ViBEModel
from dataset import get_eval_transform


# ================================================================
# PATHS
# ================================================================
QUERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/queries"
GALLERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/gallery"
MODEL_PATH = config.BEST_MODEL_PATH


# ================================================================
# DEVICE & TRANSFORM
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = get_eval_transform()


# ================================================================
# MODEL
# ================================================================
model = ViBEModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def extract_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_cloth(img)

    # Explicit normalization (safety)
    emb = F.normalize(emb, dim=1)

    return emb.cpu().numpy()[0]


def extract_clothing_id(path):
    # Example: WOMEN_Dresses_id_00004435_02_1_front.jpg
    parts = path.name.split("_")
    for i, p in enumerate(parts):
        if p == "id":
            return parts[i] + "_" + parts[i+1]
    return None


# ================================================================
# RECALL FUNCTION (NEW CLEAN VERSION)
# ================================================================
def recall_at_k(query_embs, gallery_embs, gt_indices, ks=(1, 5, 10)):
    sims = torch.matmul(query_embs, gallery_embs.T)
    results = {}

    for k in ks:
        topk = sims.topk(k=min(k, sims.size(1)), dim=1).indices
        hits = []

        for i in range(topk.size(0)):
            hits.append(int(gt_indices[i] in topk[i]))

        results[k] = sum(hits) / len(hits)

    return results


# ================================================================
# COMPUTE GALLERY EMBEDDINGS
# ================================================================
print("\nComputing gallery embeddings...")

gallery_paths = list(Path(GALLERY_DIR).glob("*.jpg"))
gallery_embeddings = []
gallery_ids = []

for p in tqdm(gallery_paths):
    emb = extract_embedding(p)
    gallery_embeddings.append(emb)
    gallery_ids.append(extract_clothing_id(p))

gallery_embeddings = torch.tensor(np.array(gallery_embeddings), dtype=torch.float32)


# ================================================================
# COMPUTE QUERY EMBEDDINGS
# ================================================================
print("\nComputing query embeddings...")

query_paths = list(Path(QUERY_DIR).glob("*.jpg"))
query_embeddings = []
gt_indices = []

for q in tqdm(query_paths):
    emb = extract_embedding(q)
    query_embeddings.append(emb)

    q_id = extract_clothing_id(q)
    gt_idx = gallery_ids.index(q_id) if q_id in gallery_ids else -1
    gt_indices.append(gt_idx)

query_embeddings = torch.tensor(np.array(query_embeddings), dtype=torch.float32)


# ================================================================
# EVALUATION
# ================================================================
print("\nRunning retrieval evaluation...")

results = recall_at_k(query_embeddings, gallery_embeddings, gt_indices)

print("\nRESULTS")
print(f"Recall@1  : {results[1]:.4f}")
print(f"Recall@5  : {results[5]:.4f}")
print(f"Recall@10 : {results[10]:.4f}")