import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import BodyClothDataset
from model import ViBEModel


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # =========================
    # DATASETS
    # =========================

    train_dataset = BodyClothDataset(
        image_dir=config.TRAIN_DIR,
        body_csv=config.BODY_VECTOR_CSV
    )

    val_dataset = BodyClothDataset(
        image_dir=config.VAL_DIR,
        body_csv=config.BODY_VECTOR_CSV
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # =========================
    # MODEL
    # =========================

    model = ViBEModel(
        body_input_dim=7,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)

    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # =========================
    # METRIC STORAGE
    # =========================

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float("inf")

    # =========================
    # TRAINING LOOP
    # =========================

    for epoch in range(config.EPOCHS):

        # -------- TRAIN --------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader)

        for body_vec, pos_img, neg_img in loop:

            body_vec = body_vec.to(device)
            pos_img = pos_img.to(device)
            neg_img = neg_img.to(device)

            optimizer.zero_grad()

            body_emb = model.encode_body(body_vec)
            pos_emb = model.encode_cloth(pos_img)
            neg_emb = model.encode_cloth(neg_img)

            loss = criterion(body_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy (pos distance < neg distance)
            pos_dist = torch.norm(body_emb - pos_emb, dim=1)
            neg_dist = torch.norm(body_emb - neg_emb, dim=1)

            correct += (pos_dist < neg_dist).sum().item()
            total += body_vec.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # -------- VALIDATION --------
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for body_vec, pos_img, neg_img in val_loader:

                body_vec = body_vec.to(device)
                pos_img = pos_img.to(device)
                neg_img = neg_img.to(device)

                body_emb = model.encode_body(body_vec)
                pos_emb = model.encode_cloth(pos_img)
                neg_emb = model.encode_cloth(neg_img)

                loss = criterion(body_emb, pos_emb, neg_emb)
                val_running_loss += loss.item()

                pos_dist = torch.norm(body_emb - pos_emb, dim=1)
                neg_dist = torch.norm(body_emb - neg_emb, dim=1)

                val_correct += (pos_dist < neg_dist).sum().item()
                val_total += body_vec.size(0)

        avg_val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total

        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # -------- SAVE BEST MODEL (LOCAL ONLY) --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "/content/best_vibe_model.pth")
            print("Best model saved locally.")

    # =========================
    # PLOT CURVES
    # =========================

    epochs = range(1, config.EPOCHS + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.show()

    # =========================
    # COPY TO DRIVE (SAFE)
    # =========================

    try:
        import shutil
        drive_path = "/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth"

        if os.path.exists("/content/best_vibe_model.pth"):
            shutil.copy("/content/best_vibe_model.pth", drive_path)
            print("Model copied to Drive successfully.")

    except Exception as e:
        print("Drive copy failed:", e)


if __name__ == "__main__":
    train()