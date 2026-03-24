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
# DEVICE & TRANSFORM (MUST MATCH TRAINING)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),           # ← Critical: same as training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# MODEL
# -----------------------------
model = ViBEModel(body_input_dim=7, embedding_dim=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# ENCODERS (use model's internal normalization)
# -----------------------------
def encode_cloth(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_cloth(img)      # already normalized inside encoder
    return emb.cpu().numpy()[0]

def encode_body(vec):
    vec = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_body(vec)
    return emb.cpu().numpy()[0]

# -----------------------------
# LOAD DATA
# -----------------------------
print("\nLoading body vectors...")
df = pd.read_csv(BODY_CSV)
cloth_paths = list(Path(TEST_DIR).glob("*.jpg"))
cloth_names = [p.name for p in cloth_paths]
df = df[df["image"].isin(cloth_names)].reset_index(drop=True)
print("Test samples:", len(df))

# -----------------------------
# COMPUTE CLOTHING EMBEDDINGS
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
# EVALUATION (500 gallery subset)
# -----------------------------
print("\nRunning subset evaluation...")
top1 = 0
top5 = 0
n = len(df)

for _, row in tqdm(df.iterrows(), total=n):
    body_vec = row.values[1:].astype(np.float32)
    body_emb = encode_body(body_vec)
    gt = row["image"]

    # Random 500 gallery (realistic test)
    gallery_idx = random.sample(range(len(cloth_embeddings)), 499)
    gt_idx = cloth_images.index(gt)
    if gt_idx not in gallery_idx:
        gallery_idx.append(gt_idx)

    gallery_emb = cloth_embeddings[gallery_idx]
    gallery_names = [cloth_images[i] for i in gallery_idx]

    scores = np.dot(gallery_emb, body_emb)
    idx = np.argsort(-scores)
    ranked = [gallery_names[i] for i in idx]

    if gt in ranked[:1]:
        top1 += 1
    if gt in ranked[:5]:
        top5 += 1

print("\n" + "="*50)
print("FASHIONISTA (500 GALLERY)")
print("="*50)
print(f"Top-1 Accuracy : {top1/n:.4f}")
print(f"Top-5 Accuracy : {top5/n:.4f}")