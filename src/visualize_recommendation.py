import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import ViBEModel

# PATHS
MODEL_PATH = "/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth"
TEST_DIR = "/content/drive/MyDrive/SmartWardrobe/body_shape_recommendor/data/split/test"
BODY_CSV = "/content/drive/MyDrive/SmartWardrobe/body_shape_recommendor/data/body_vectors.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAME TRANSFORM AS TRAINING
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),           # ← CRITICAL
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = ViBEModel(body_input_dim=7, embedding_dim=128).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def encode_cloth(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_cloth(img)          # already normalized inside encoder
    return emb.cpu().numpy()[0]

def encode_body(vec):
    vec = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_body(vec)
    return emb.cpu().numpy()[0]

# Load data
df = pd.read_csv(BODY_CSV)
cloth_paths = list(Path(TEST_DIR).glob("*.jpg"))
cloth_names = [p.name for p in cloth_paths]
df = df[df["image"].isin(cloth_names)].reset_index(drop=True)
print("Test samples:", len(df))

# Precompute
cloth_embeddings = []
cloth_images = []
print("\nComputing clothing embeddings...")
for p in tqdm(cloth_paths):
    emb = encode_cloth(p)
    cloth_embeddings.append(emb)
    cloth_images.append(p)
cloth_embeddings = np.array(cloth_embeddings)

print("\nShowing recommendations...\n")
for i in range(min(8, len(df))):          # show more examples
    row = df.iloc[i]
    body_vec = row.values[1:].astype(np.float32)
    body_emb = encode_body(body_vec)
    gt_name = row["image"]
    gt_idx = cloth_images.index(Path(TEST_DIR) / gt_name)

    # Random realistic gallery
    gallery_idx = list(range(len(cloth_embeddings)))
    random.shuffle(gallery_idx)
    gallery_idx = gallery_idx[:50]
    if gt_idx not in gallery_idx:
        gallery_idx.append(gt_idx)

    gallery_emb = cloth_embeddings[gallery_idx]
    gallery_imgs = [cloth_images[j] for j in gallery_idx]

    scores = np.dot(gallery_emb, body_emb)
    idx = np.argsort(-scores)[:5]
    retrieved = [gallery_imgs[j] for j in idx]

    # Plot
    plt.figure(figsize=(18, 3))
    plt.subplot(1, 6, 1)
    plt.imshow(Image.open(Path(TEST_DIR) / gt_name))
    plt.title("Ground Truth")
    plt.axis("off")
    for j, img_path in enumerate(retrieved):
        plt.subplot(1, 6, j+2)
        plt.imshow(Image.open(img_path))
        plt.title(f"Top {j+1}")
        plt.axis("off")
    plt.show()