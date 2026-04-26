# evaluate.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

import config
from model import ViBEModel
from dataset import get_eval_transform
from polyvore_dataset import PolyvoreDataset


# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    model = ViBEModel().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()
    return model


# ============================================================
# DEEPFASHION EVALUATION
# ============================================================
@torch.no_grad()
def evaluate_deepfashion(model):
    print("\n" + "="*55)
    print("  DeepFashion Retrieval Evaluation")
    print("="*55)

    QUERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/queries"
    GALLERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/gallery"

    transform = get_eval_transform()

    def extract_embedding(img_path):
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        emb = model.encode_cloth(img)
        return emb.cpu().numpy()[0]

    def extract_id(path):
        parts = path.name.split("_")
        for i, p in enumerate(parts):
            if p == "id":
                return parts[i] + "_" + parts[i+1]
        return None

    gallery_paths = list(Path(GALLERY_DIR).glob("*.jpg"))

    print("  Encoding gallery...")
    gallery_embs = []
    gallery_ids = []

    for p in tqdm(gallery_paths):
        gallery_embs.append(extract_embedding(p))
        gallery_ids.append(extract_id(p))

    gallery_embs = np.array(gallery_embs)

    query_paths = list(Path(QUERY_DIR).glob("*.jpg"))

    r1, r5, r10 = 0, 0, 0
    n = len(query_paths)

    print("  Evaluating queries...")
    for q in tqdm(query_paths):
        q_emb = extract_embedding(q)
        q_id = extract_id(q)

        scores = np.dot(gallery_embs, q_emb)
        idx = np.argsort(-scores)
        ranked_ids = [gallery_ids[i] for i in idx]

        if q_id in ranked_ids[:1]:
            r1 += 1
        if q_id in ranked_ids[:5]:
            r5 += 1
        if q_id in ranked_ids[:10]:
            r10 += 1

    print("\nResults:")
    print(f"  Recall@1  : {r1/n:.4f}")
    print(f"  Recall@5  : {r5/n:.4f}")
    print(f"  Recall@10 : {r10/n:.4f}")


# ============================================================
# POLYVORE EVALUATION (YOUR DATASET)
# ============================================================
@torch.no_grad()
def evaluate_polyvore(model, sample_size=5000):
    print("\n" + "="*55)
    print("  Polyvore Compatibility Evaluation")
    print("="*55)

    dataset = PolyvoreDataset(
        arrow_dir=config.POLYVORE_ARROW_DIR,
        transform=get_eval_transform()
    )

    model.eval()

    # sample subset (important)
    if len(dataset) > sample_size:
        indices = torch.randperm(len(dataset))[:sample_size]
    else:
        indices = torch.arange(len(dataset))

    print(f"  Using {len(indices)} samples")

    all_embs = []
    outfit_ids = []

    print("  Encoding items...")
    for idx in tqdm(indices):
        img, outfit_id, _ = dataset[int(idx)]

        img = img.unsqueeze(0).to(device)
        emb = model.encode_cloth(img)
        emb = F.normalize(emb, dim=1)

        all_embs.append(emb.cpu())
        outfit_ids.append(outfit_id)

    all_embs = torch.cat(all_embs, dim=0)

    recalls = {1: 0, 5: 0, 10: 0}
    N = all_embs.size(0)

    print("  Computing retrieval...")

    for i in tqdm(range(N)):
        query = all_embs[i].unsqueeze(0)
        sims = torch.matmul(query, all_embs.T).squeeze(0)

        sims[i] = -1  # remove self

        ranked = torch.argsort(sims, descending=True)
        gt = outfit_ids[i]

        for k in recalls.keys():
            topk = ranked[:k]
            if any(outfit_ids[j] == gt for j in topk):
                recalls[k] += 1

    print("\nResults:")
    for k in recalls:
        print(f"  Recall@{k}: {recalls[k]/N:.4f}")




@torch.no_grad()
def evaluate_fashionista_gallery(model, device, gallery_size=500, seed=42):
    print("\n" + "="*60)
    print("  Fashionista Gallery Retrieval")
    print("="*60)

    rng = random.Random(seed)

    df = pd.read_csv(config.BODY_VECTOR_CSV)
    cloth_paths = list(Path(config.TEST_DIR).glob("*.jpg"))
    cloth_names = [p.name for p in cloth_paths]

    df = df[df["image"].isin(cloth_names)].reset_index(drop=True)
    print(f"  Test samples: {len(df)}")

    print("  Computing clothing embeddings...")
    cloth_embs = []
    for p in tqdm(cloth_paths):
        cloth_embs.append(encode_cloth(model, p, device))
    cloth_embs = np.array(cloth_embs)

    recall1 = recall5 = recall10 = 0
    mrr_vals = []
    valid = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        gt = row["image"]
        if gt not in cloth_names:
            continue

        body_vec = row.values[1:].astype(np.float32)
        body_emb = encode_body(model, body_vec, device)

        gt_idx = cloth_names.index(gt)

        all_idx = list(range(len(cloth_embs)))
        all_idx.remove(gt_idx)

        sampled = rng.sample(all_idx, min(gallery_size - 1, len(all_idx)))
        gallery_idx = sampled + [gt_idx]

        gallery_emb = cloth_embs[gallery_idx]
        gallery_names = [cloth_names[i] for i in gallery_idx]

        scores = np.dot(gallery_emb, body_emb)
        ranked = np.argsort(-scores)
        ranked_names = [gallery_names[i] for i in ranked]

        valid += 1
        if gt in ranked_names[:1]:
            recall1 += 1
        if gt in ranked_names[:5]:
            recall5 += 1
        if gt in ranked_names[:10]:
            recall10 += 1

        rank = ranked_names.index(gt) + 1
        mrr_vals.append(1.0 / rank)

    print(f"\n  Gallery size: {gallery_size}")
    print(f"  Recall@1  : {recall1 / max(valid,1):.4f}")
    print(f"  Recall@5  : {recall5 / max(valid,1):.4f}")
    print(f"  Recall@10 : {recall10 / max(valid,1):.4f}")
    print(f"  MRR       : {np.mean(mrr_vals):.4f}")
    
    
    
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["deepfashion", "polyvore", "all"],
        default="all"
    )
    args = parser.parse_args()

    model = load_model()

    if args.mode in ["deepfashion", "all"]:
        evaluate_deepfashion(model)

    if args.mode in ["polyvore", "all"]:
        evaluate_polyvore(model)