import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from model import ViBEModel


# -----------------------------
# PATHS
# -----------------------------

QUERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/queries"
GALLERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/gallery"

MODEL_PATH = "/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth"


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
# ENCODER
# -----------------------------

def encode_image(path):

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_cloth(img)

    emb = emb.cpu().numpy()[0]

    # normalize embedding
    emb = emb / np.linalg.norm(emb)

    return emb


# -----------------------------
# EXTRACT CLOTHING ID
# -----------------------------

def extract_clothing_id(path):

    for part in path.parts:
        if part.startswith("id_"):
            return part

    return None


# -----------------------------
# LOAD GALLERY
# -----------------------------

print("\nComputing gallery embeddings...")

gallery_paths = sorted(list(Path(GALLERY_DIR).glob("*.jpg")))

gallery_embeddings = []
gallery_ids = []

for p in tqdm(gallery_paths):

    emb = encode_image(p)

    gallery_embeddings.append(emb)
    gallery_ids.append(extract_clothing_id(p))

gallery_embeddings = np.array(gallery_embeddings)


# -----------------------------
# EVALUATION
# -----------------------------

print("\nRunning retrieval evaluation...")

query_paths = sorted(list(Path(QUERY_DIR).glob("*.jpg")))

recall1 = 0
recall5 = 0
recall10 = 0


for q in tqdm(query_paths):

    q_emb = encode_image(q)

    scores = gallery_embeddings @ q_emb

    idx = np.argsort(-scores)

    # remove self-match
    idx = [i for i in idx if gallery_paths[i].name != q.name]

    retrieved_ids = [gallery_ids[i] for i in idx]

    gt_id = extract_clothing_id(q)

    if gt_id in retrieved_ids[:1]:
        recall1 += 1

    if gt_id in retrieved_ids[:5]:
        recall5 += 1

    if gt_id in retrieved_ids[:10]:
        recall10 += 1


n = len(query_paths)

print("\n====================")
print("DEEPFASHION RESULTS")
print("====================")

print("Recall@1 :", round(recall1/n,4))
print("Recall@5 :", round(recall5/n,4))
print("Recall@10:", round(recall10/n,4))