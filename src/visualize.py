import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

from model import ViBEModel

QUERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/queries"
GALLERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/gallery"

MODEL_PATH = "/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth"

def extract_clothing_id(path):

    parts = path.name.split("_")

    for i,p in enumerate(parts):
        if p == "id":
            return parts[i] + "_" + parts[i+1]

    return "unknown"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

model = ViBEModel(
    body_input_dim=7,
    embedding_dim=128
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def extract_embedding(img_path):

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_cloth(img)

    emb = emb.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)

    return emb


print("Computing gallery embeddings...")

gallery_paths = list(Path(GALLERY_DIR).glob("*.jpg"))

gallery_embeddings = []
gallery_names = []

for p in tqdm(gallery_paths):

    emb = extract_embedding(p)

    gallery_embeddings.append(emb)
    gallery_names.append(p)

gallery_embeddings = np.array(gallery_embeddings)


print("Running visual retrieval...")

query_paths = list(Path(QUERY_DIR).glob("*.jpg"))

for q in query_paths[:5]:

    q_emb = extract_embedding(q)

    dists = np.linalg.norm(gallery_embeddings - q_emb, axis=1)

    idx = np.argsort(dists)[:5]

    retrieved = [gallery_names[i] for i in idx]

    plt.figure(figsize=(15,3))

    plt.subplot(1,6,1)
    plt.imshow(Image.open(q))
    plt.title(extract_clothing_id(q))
    plt.axis("off")

    for i,img in enumerate(retrieved):

        plt.subplot(1,6,i+2)
        plt.imshow(Image.open(img))
        plt.title(extract_clothing_id(img))
        plt.axis("off")

    plt.show()