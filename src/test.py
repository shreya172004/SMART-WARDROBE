import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

from model import ViBEModel

# paths
QUERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/queries"
GALLERY_DIR = "/content/drive/MyDrive/SmartWardrobe/deepfashion_test_subset/gallery"

MODEL_PATH = "/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# load model
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

    return emb.cpu().numpy()[0]


print("Computing gallery embeddings...")

gallery_paths = list(Path(GALLERY_DIR).glob("*.jpg"))

gallery_embeddings = []
gallery_names = []

for p in tqdm(gallery_paths):

    emb = extract_embedding(p)

    gallery_embeddings.append(emb)
    gallery_names.append(p.name)

gallery_embeddings = np.array(gallery_embeddings)


print("Running retrieval evaluation...")

query_paths = list(Path(QUERY_DIR).glob("*.jpg"))

recall1 = 0
recall5 = 0
recall10 = 0

for q in tqdm(query_paths):

    q_emb = extract_embedding(q)

    # compute distance
    dists = np.linalg.norm(gallery_embeddings - q_emb, axis=1)

    idx = np.argsort(dists)

    ranked = [gallery_names[i] for i in idx]

    qname = q.name

    if qname in ranked[:1]:
        recall1 += 1

    if qname in ranked[:5]:
        recall5 += 1

    if qname in ranked[:10]:
        recall10 += 1


n = len(query_paths)

print("\nRESULTS")
print("Recall@1 :", recall1/n)
print("Recall@5 :", recall5/n)
print("Recall@10:", recall10/n)