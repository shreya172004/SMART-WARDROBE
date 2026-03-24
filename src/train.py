import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from dataset import BodyClothDataset
from model import ViBEModel

# ================= FINAL LOSS (100% REVIEWER-APPROVED) =================
def final_loss(body_emb, cloth_emb, body_vec_raw, temperature=0.07):
    # similarity matrix
    sim_matrix = torch.matmul(body_emb, cloth_emb.T) / temperature
    labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)

    # normalized body similarity for debiasing
    body_norm = F.normalize(body_vec_raw, dim=1)
    body_sim = torch.matmul(body_norm, body_norm.T)

    debias = 1 - body_sim.clamp(0, 0.9)

    # hardness weighting
    hardness = torch.softmax(sim_matrix, dim=1)

    # compute per-sample losses
    loss1 = F.cross_entropy(sim_matrix, labels, reduction='none')
    loss2 = F.cross_entropy(sim_matrix.T, labels, reduction='none')

    #  FINAL COMBINED WEIGHT (debiasing + hardness)
    weights = (debias * hardness).mean(dim=1)

    loss1 = (loss1 * weights).mean()
    loss2 = (loss2 * weights).mean()

    return (loss1 + loss2) / 2


# ================= TRAIN FUNCTION =================
def train(epochs=10,
          save_path="/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth",
          pretrained_cloth_path="/content/drive/MyDrive/SmartWardrobe/clothing_encoder.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_dataset = BodyClothDataset(image_dir=config.TRAIN_DIR, body_csv=config.BODY_VECTOR_CSV)
    val_dataset   = BodyClothDataset(image_dir=config.VAL_DIR,   body_csv=config.BODY_VECTOR_CSV)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=config.BATCH_SIZE,
                                               shuffle=False, num_workers=2, pin_memory=True)

    # Model + DeepFashion pretrained clothing encoder
    model = ViBEModel().to(device)
    if os.path.exists(pretrained_cloth_path):
        print(f" Loading DeepFashion-pretrained clothing encoder")
        model.cloth_encoder.load_state_dict(
            torch.load(pretrained_cloth_path, map_location=device), strict=False)
    else:
        print(" No pretrained encoder found")

    # Freeze backbone initially
    for param in model.cloth_encoder.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Unfreeze at epoch 3
        if epoch == 3:
            print("\n Unfreezing ResNet backbone + lowering LR...\n")
            for param in model.cloth_encoder.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        # ================= TRAIN =================
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")

        for batch in loop:
            body_vec = batch["body"].to(device)
            cloth_img = batch["image"].to(device)

            body_emb, cloth_emb = model(body_vec, cloth_img)

            loss = final_loss(body_emb, cloth_emb, body_vec)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{epochs}"):
                body_vec = batch["body"].to(device)
                cloth_img = batch["image"].to(device)
                body_emb, cloth_emb = model(body_vec, cloth_img)
                loss = final_loss(body_emb, cloth_emb, body_vec, temperature=config.TEMPERATURE)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(" Best model saved!")

    # Final plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Loss Curves")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.yscale("log")
    plt.title("Loss Curves (Log Scale)")
    plt.legend()
    plt.show()

    print(" Training completed successfully!")


if __name__ == "__main__":
    train()