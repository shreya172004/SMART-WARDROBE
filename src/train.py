import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BodyClothDataset
from model import ViBEModel
import config


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Dataset
    train_dataset = BodyClothDataset(
        image_dir=config.TRAIN_DIR,
        body_csv=config.BODY_VECTOR_CSV
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    print(f"Training samples: {len(train_dataset)}")

    # Model
    model = ViBEModel(
        body_input_dim=7,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)

    # Loss (Triplet Loss)
    criterion = nn.TripletMarginLoss(margin=1.0)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR
    )

    # Training Loop
    for epoch in range(config.EPOCHS):

        model.train()
        total_loss = 0

        loop = tqdm(train_loader)

        for body_vec, pos_img, neg_img in loop:

            body_vec = body_vec.to(device)
            pos_img = pos_img.to(device)
            neg_img = neg_img.to(device)

            # Forward pass
            body_emb = model.encode_body(body_vec)
            pos_emb = model.encode_cloth(pos_img)
            neg_emb = model.encode_cloth(neg_img)

            # Triplet loss
            loss = criterion(body_emb, pos_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{config.EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "vibe_model.pth")
    print("\nModel saved as vibe_model.pth")


if __name__ == "__main__":
    train()