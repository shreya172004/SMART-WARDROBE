import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from deepfashion_dataset import DeepFashionDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= MODEL =================
class ClothingEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        x = F.normalize(x, dim=1)
        return x


# ================= LOSS =================
def contrastive_loss(embeddings, labels, temperature=0.07):

    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()

    exp_sim = torch.exp(sim_matrix)

    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

    loss = -mean_log_prob_pos.mean()

    return loss


# ================= TRAIN =================
def train():

    dataset = DeepFashionDataset(
        root_dir = "/content/drive/MyDrive/SmartWardrobe/body_shape_recommendor/data/deepfashion_subset"
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ClothingEncoder().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):

        total_loss = 0

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            embeddings = model(images)

            loss = contrastive_loss(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "clothing_encoder.pth")
    print("✅ Model saved!")


if __name__ == "__main__":
    train()