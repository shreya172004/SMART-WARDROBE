import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
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

print("\nLoading body vectors...")

df = pd.read_csv(BODY_CSV)

cloth_paths = list(Path(TEST_DIR).glob("*.jpg"))

cloth_names = [p.name for p in cloth_paths]

df = df[df["image"].isin(cloth_names)]

df = df.reset_index(drop=True)

print("Test samples:", len(df))


# -----------------------------
# COMPUTE CLOTH EMBEDDINGS
# -----------------------------

print("\nComputing clothing embeddings...")

cloth_embeddings = []
cloth_images = []

for p in tqdm(cloth_paths):

    emb = encode_cloth(p)

    cloth_embeddings.append(emb)
    cloth_images.append(p.name)

cloth_embeddings = np.array(cloth_embeddings)


# -----------------------------
# EVALUATION
# -----------------------------

print("\nRunning subset evaluation...")

top1 = 0
top5 = 0


for _,row in tqdm(df.iterrows(), total=len(df)):

    body_vec = row.values[1:].astype(np.float32)

    body_emb = encode_body(body_vec)

    gt = row["image"]

    gt_idx = cloth_images.index(gt)

    # choose random gallery of 500
    gallery_idx = random.sample(range(len(cloth_embeddings)), 499)

    if gt_idx not in gallery_idx:
        gallery_idx.append(gt_idx)

    gallery_emb = cloth_embeddings[gallery_idx]
    gallery_names = [cloth_images[i] for i in gallery_idx]

    scores = gallery_emb @ body_emb

    idx = np.argsort(-scores)

    ranked = [gallery_names[i] for i in idx]

    if gt in ranked[:1]:
        top1 += 1

    if gt in ranked[:5]:
        top5 += 1


n = len(df)

print("\n====================")
print("FASHIONISTA (500 GALLERY)")
print("====================")

print("Top1 Accuracy :", round(top1/n,4))
print("Top5 Accuracy :", round(top5/n,4))