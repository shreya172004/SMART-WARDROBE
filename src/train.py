import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from dataset import BodyClothDataset
from model import ViBEModel
import config


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =============================
    # DATASETS
    # =============================

    train_dataset = BodyClothDataset(
        image_dir=config.TRAIN_DIR,
        body_csv=config.BODY_VECTOR_CSV
    )

    val_dataset = BodyClothDataset(
        image_dir=config.VAL_DIR,
        body_csv=config.BODY_VECTOR_CSV
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              num_workers=2)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            num_workers=2)

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # =============================
    # MODEL
    # =============================

    model = ViBEModel(
        body_input_dim=7,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)

    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # =============================
    # METRIC STORAGE
    # =============================

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float("inf")

    # =============================
    # TRAINING LOOP
    # =============================

    for epoch in range(config.EPOCHS):

        # ---- TRAIN ----
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        loop = tqdm(train_loader)

        for body_vec, pos_img, neg_img in loop:

            body_vec = body_vec.to(device)
            pos_img = pos_img.to(device)
            neg_img = neg_img.to(device)

            body_emb = model.encode_body(body_vec)
            pos_emb = model.encode_cloth(pos_img)
            neg_emb = model.encode_cloth(neg_img)

            loss = criterion(body_emb, pos_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Triplet accuracy
            pos_dist = torch.norm(body_emb - pos_emb, dim=1)
            neg_dist = torch.norm(body_emb - neg_emb, dim=1)

            correct_train += (pos_dist < neg_dist).sum().item()
            total_train += body_vec.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)

        # ---- VALIDATION ----
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for body_vec, pos_img, neg_img in val_loader:

                body_vec = body_vec.to(device)
                pos_img = pos_img.to(device)
                neg_img = neg_img.to(device)

                body_emb = model.encode_body(body_vec)
                pos_emb = model.encode_cloth(pos_img)
                neg_emb = model.encode_cloth(neg_img)

                loss = criterion(body_emb, pos_emb, neg_emb)
                total_val_loss += loss.item()

                pos_dist = torch.norm(body_emb - pos_emb, dim=1)
                neg_dist = torch.norm(body_emb - neg_emb, dim=1)

                correct_val += (pos_dist < neg_dist).sum().item()
                total_val += body_vec.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)

        print(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_accuracy:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                "/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth"
            )
            print("Best model saved!")

    # =============================
    # PLOT CURVES
    # =============================

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.savefig("/content/drive/MyDrive/SmartWardrobe/training_curves.png")
    plt.show()

    print("Training complete.")


if __name__ == "__main__":
    train()