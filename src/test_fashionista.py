import torch
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

from model import ViBEModel


MODEL_PATH = "/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth"

TEST_DIR = "/content/drive/MyDrive/SmartWardrobe/body_shape_recommendor/data/split/test"
BODY_CSV = "/content/drive/MyDrive/SmartWardrobe/body_shape_recommendor/data/body_vectors.csv"


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


def encode_cloth(path):

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_cloth(img)

    emb = emb.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)

    return emb


def encode_body(vec):

    vec = torch.tensor(vec).float().unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_body(vec)

    emb = emb.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)

    return emb


print("Loading body vectors...")

df = pd.read_csv(BODY_CSV)


print("Computing clothing embeddings...")

cloth_paths = list(Path(TEST_DIR).glob("*.jpg"))

cloth_embeddings = []
cloth_names = []

for p in tqdm(cloth_paths):

    emb = encode_cloth(p)

    cloth_embeddings.append(emb)
    cloth_names.append(p.name)

cloth_embeddings = np.array(cloth_embeddings)


print("Running recommendation evaluation...")

top1 = 0
top5 = 0

for _,row in tqdm(df.iterrows(), total=len(df)):

    body_vec = row.values[1:]

    body_emb = encode_body(body_vec)

    dists = -np.dot(cloth_embeddings, body_emb)

    idx = np.argsort(dists)

    ranked = [cloth_names[i] for i in idx]

    correct_item = row["image"]

    if correct_item in ranked[:1]:
        top1 += 1

    if correct_item in ranked[:5]:
        top5 += 1


n = len(df)

print("\nRESULTS")
print("Top1 Accuracy:", top1/n)
print("Top5 Accuracy:", top5/n)