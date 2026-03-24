import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

from model import ViBEModel


# -----------------------------
# PATHS
# -----------------------------
MODEL_PATH = "/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth"
TEST_DIR = "/content/drive/MyDrive/SmartWardrobe/body_shape_recommendor/data/split/test"
BODY_CSV = "/content/drive/MyDrive/SmartWardrobe/body_shape_recommendor/data/body_vectors.csv"


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# -----------------------------
# MODEL
# -----------------------------
model = ViBEModel(
    body_input_dim=7,
    embedding_dim=128
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# -----------------------------
# ENCODERS
# -----------------------------
def encode_cloth(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_cloth(img)

    emb = emb.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)
    return emb


def encode_body(vec):
    vec = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_body(vec)

    emb = emb.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)
    return emb


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(BODY_CSV)

cloth_paths = list(Path(TEST_DIR).glob("*.jpg"))
cloth_names = [p.name for p in cloth_paths]

df = df[df["image"].isin(cloth_names)].reset_index(drop=True)

print("Test samples:", len(df))


# -----------------------------
# PRECOMPUTE EMBEDDINGS
# -----------------------------
cloth_embeddings = []
cloth_images = []

print("\nComputing clothing embeddings...")

for p in tqdm(cloth_paths):
    emb = encode_cloth(p)
    cloth_embeddings.append(emb)
    cloth_images.append(p)

cloth_embeddings = np.array(cloth_embeddings)


# -----------------------------
# VISUALIZATION
# -----------------------------
print("\nShowing recommendations...\n")

num_examples = min(5, len(df))

for i in range(num_examples):

    row = df.iloc[i]

    body_vec = row.values[1:].astype(np.float32)
    body_emb = encode_body(body_vec)

    gt_name = row["image"]
    gt_idx = cloth_images.index(Path(TEST_DIR) / gt_name)

    # -----------------------------
    # RANDOM GALLERY (REALISTIC)
    # -----------------------------
    gallery_idx = random.sample(range(len(cloth_embeddings)), 49)

    if gt_idx not in gallery_idx:
        gallery_idx.append(gt_idx)

    gallery_emb = cloth_embeddings[gallery_idx]
    gallery_imgs = [cloth_images[j] for j in gallery_idx]

    scores = np.dot(gallery_emb, body_emb)
    idx = np.argsort(-scores)[:5]

    retrieved = [gallery_imgs[j] for j in idx]

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure(figsize=(18,3))

    # Ground Truth
    plt.subplot(1,6,1)
    plt.imshow(Image.open(Path(TEST_DIR) / gt_name))
    plt.title("GT")
    plt.axis("off")

    # Predictions
    for j, img_path in enumerate(retrieved):

        plt.subplot(1,6,j+2)

        img = Image.open(img_path)
        plt.imshow(img)

        if img_path.name == gt_name:
            plt.title(f"Top {j+1} ✓", color="green")
        else:
            plt.title(f"Top {j+1}")

        plt.axis("off")

    plt.show()