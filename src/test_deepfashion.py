import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

from model import ViBEModel


QUERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/queries"
GALLERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/gallery"

MODEL_PATH = "/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth"


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


def extract_clothing_id(path):

    # example filename
    # WOMEN_Dresses_id_00004435_02_1_front.jpg

    parts = path.name.split("_")

    for i,p in enumerate(parts):
        if p == "id":
            return parts[i] + "_" + parts[i+1]

    return None


print("Computing gallery embeddings...")

gallery_paths = list(Path(GALLERY_DIR).glob("*.jpg"))

gallery_embeddings = []
gallery_ids = []

for p in tqdm(gallery_paths):

    emb = extract_embedding(p)

    gallery_embeddings.append(emb)
    gallery_ids.append(extract_clothing_id(p))

gallery_embeddings = np.array(gallery_embeddings)


print("Running retrieval evaluation...")

query_paths = list(Path(QUERY_DIR).glob("*.jpg"))

recall1 = 0
recall5 = 0
recall10 = 0

for q in tqdm(query_paths):

    q_emb = extract_embedding(q)

    q_id = extract_clothing_id(q)

    scores = np.dot(gallery_embeddings, q_emb)
    idx = np.argsort(-scores)

    ranked_ids = [gallery_ids[i] for i in idx]

    if q_id in ranked_ids[:1]:
        recall1 += 1

    if q_id in ranked_ids[:5]:
        recall5 += 1

    if q_id in ranked_ids[:10]:
        recall10 += 1


n = len(query_paths)

print("\nRESULTS")
print("Recall@1 :", recall1/n)
print("Recall@5 :", recall5/n)
print("Recall@10:", recall10/n)