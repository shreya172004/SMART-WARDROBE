import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import BodyClothDataset
from model import ViBEModel


# =========================
# PATHS (DIRECT TO DRIVE)
# =========================

CHECKPOINT_PATH = "/content/drive/MyDrive/SmartWardrobe/checkpoint.pth"
BEST_MODEL_PATH = "/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth"


def save_checkpoint(epoch, batch_idx, model, optimizer,
                    train_losses, val_losses,
                    train_accs, val_accs, best_val_loss):

    torch.save({
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_loss": best_val_loss
    }, CHECKPOINT_PATH)


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
    # RESUME LOGIC
    # =========================

    start_epoch = 0
    start_batch = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")

    if os.path.exists(CHECKPOINT_PATH):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint.get("batch_idx", 0)

        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        train_accs = checkpoint["train_accs"]
        val_accs = checkpoint["val_accs"]
        best_val_loss = checkpoint["best_val_loss"]

        print(f"Resumed from epoch {start_epoch}, batch {start_batch}")

    # =========================
    # TRAIN LOOP
    # =========================

    for epoch in range(start_epoch, config.EPOCHS):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Train Epoch {epoch+1}")

        for batch_idx, (body_vec, pos_img, neg_img) in loop:

            if epoch == start_epoch and batch_idx < start_batch:
                continue  # skip already processed batches

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

            pos_dist = torch.norm(body_emb - pos_emb, dim=1)
            neg_dist = torch.norm(body_emb - neg_emb, dim=1)

            correct += (pos_dist < neg_dist).sum().item()
            total += body_vec.size(0)

            # 🔥 SAVE MID-EPOCH EVERY 200 BATCHES
            if batch_idx % 200 == 0 and batch_idx != 0:
                save_checkpoint(epoch, batch_idx, model, optimizer,
                                train_losses, val_losses,
                                train_accs, val_accs, best_val_loss)
                print(f"Mid-epoch checkpoint saved at batch {batch_idx}")

        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # =========================
        # VALIDATION WITH TQDM
        # =========================

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for body_vec, pos_img, neg_img in val_loop:

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

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # 🔥 SAVE BEST MODEL DIRECTLY TO DRIVE
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Best model updated (Drive).")

        # 🔥 SAVE CHECKPOINT AFTER EVERY EPOCH
        save_checkpoint(epoch + 1, 0, model, optimizer,
                        train_losses, val_losses,
                        train_accs, val_accs, best_val_loss)

        print("Epoch checkpoint saved.")

    # =========================
    # PLOT CURVES
    # =========================

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label="Train")
    plt.plot(epochs, val_accs, label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    train()