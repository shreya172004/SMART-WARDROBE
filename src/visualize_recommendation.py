import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

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
cloth_images = []

for p in tqdm(cloth_paths):

    emb = encode_cloth(p)

    cloth_embeddings.append(emb)
    cloth_images.append(p)

cloth_embeddings = np.array(cloth_embeddings)


print("Showing recommendations...")

for i in range(5):

    row = df.iloc[i]

    body_vec = row.values[1:]

    body_emb = encode_body(body_vec)

    scores = -np.dot(cloth_embeddings, body_emb)

    idx = np.argsort(scores)[:5]

    retrieved = [cloth_images[i] for i in idx]

    plt.figure(figsize=(15,3))

    for j,img in enumerate(retrieved):

        plt.subplot(1,5,j+1)
        plt.imshow(Image.open(img))
        plt.axis("off")
        plt.title(f"Top {j+1}")

    plt.show()