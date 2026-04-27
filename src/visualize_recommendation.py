# visualize_recommendation.py — FINAL CLEAN VERSION

import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

import config
from model import ViBEModel
from dataset import get_eval_transform
from loss import check_embedding_collapse


# ================================================================
# SETUP
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViBEModel(
    body_input_dim=config.BODY_INPUT_DIM,
    embedding_dim=config.EMBEDDING_DIM
).to(device)

model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
model.eval()

transform = get_eval_transform()


# ================================================================
# LOAD BODY STATS (🔥 IMPORTANT FIX)
# ================================================================

df_full = pd.read_csv(config.BODY_VECTOR_CSV)
body_matrix = np.stack(df_full.iloc[:, 1:].values.astype(np.float32))

body_mean = body_matrix.mean(axis=0)
body_std = body_matrix.std(axis=0)
body_std[body_std < 1e-8] = 1.0


# ================================================================
# ENCODERS (🔥 FIXED)
# ================================================================

@torch.no_grad()
def encode_cloth(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    emb = model.encode_cloth(img)
    return emb.cpu().numpy()[0]


@torch.no_grad()
def encode_body(vec):
    vec = (vec - body_mean) / (body_std + 1e-8)   # 🔥 CRITICAL FIX

    t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    emb = model.encode_body(t)

    return emb.cpu().numpy()[0]


# ================================================================
# LOAD DATA
# ================================================================

df = pd.read_csv(config.BODY_VECTOR_CSV)

cloth_paths = list(Path(config.TEST_DIR).glob("*.jpg"))
cloth_names = [p.name for p in cloth_paths]

df = df[df["image"].isin(cloth_names)].reset_index(drop=True)

print(f"Test samples: {len(df)}")


# ================================================================
# PRECOMPUTE CLOTHING EMBEDDINGS
# ================================================================

print("Computing clothing embeddings...")
cloth_embeddings = []

for p in tqdm(cloth_paths):
    cloth_embeddings.append(encode_cloth(p))

cloth_embeddings = np.array(cloth_embeddings)


# ================================================================
# EMBEDDING HEALTH CHECK
# ================================================================

print("\nEmbedding health check:")
check_embedding_collapse(
    torch.tensor(cloth_embeddings),
    torch.tensor(cloth_embeddings)
)


# ================================================================
# VISUALIZATION
# ================================================================

def show_recommendations(n_examples=6, gallery_size=200):
    rows = min(n_examples, len(df))

    for i in range(rows):
        row = df.iloc[i]

        body_vec = row.values[1:].astype(np.float32)
        body_emb = encode_body(body_vec)
        gt_name = row["image"]

        # -------------------------------
        # Build gallery
        # -------------------------------
        gt_idx = cloth_names.index(gt_name)

        all_idx = list(range(len(cloth_embeddings)))
        all_idx.remove(gt_idx)

        sampled = random.sample(all_idx, min(gallery_size - 1, len(all_idx)))
        gallery_idx = sampled + [gt_idx]

        gallery_emb = cloth_embeddings[gallery_idx]
        gallery_imgs = [cloth_paths[j] for j in gallery_idx]

        # -------------------------------
        # Compute similarity
        # -------------------------------
        scores = np.dot(gallery_emb, body_emb)
        top5 = np.argsort(-scores)[:5]

        retrieved = [(gallery_imgs[j], scores[j]) for j in top5]

        # -------------------------------
        # Plot
        # -------------------------------
        fig, axes = plt.subplots(1, 6, figsize=(18, 3))

        # Ground truth
        ax = axes[0]
        ax.imshow(Image.open(Path(config.TEST_DIR) / gt_name))
        ax.set_title("Ground Truth", fontsize=9)
        ax.axis("off")

        # Top 5
        for j, (img_path, score) in enumerate(retrieved):
            ax = axes[j + 1]
            ax.imshow(Image.open(img_path))

            is_gt = (img_path.name == gt_name)
            color = "green" if is_gt else "black"

            ax.set_title(f"Top {j+1}\nsim={score:.3f}", fontsize=8, color=color)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            f"/content/drive/MyDrive/SmartWardrobe/viz_sample_{i}.png",
            dpi=100,
            bbox_inches="tight"
        )
        plt.show()
        plt.close()


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    show_recommendations(n_examples=8, gallery_size=200)