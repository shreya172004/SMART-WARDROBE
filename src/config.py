import os

BASE_DIR = "/content/drive/MyDrive/SmartWardrobe/body_shape_recommendor/data"

TRAIN_DIR = os.path.join(BASE_DIR, "split/train")
VAL_DIR   = os.path.join(BASE_DIR, "split/val")
TEST_DIR  = os.path.join(BASE_DIR, "split/test")

BODY_VECTOR_CSV = os.path.join(BASE_DIR, "body_vectors.csv")

# Image + model settings
IMAGE_SIZE = 224
EMBEDDING_DIM = 128

# 🔥 IMPORTANT CHANGE
BATCH_SIZE = 32   # was 16 → increase for contrastive learning

# Training
LR = 1e-4
EPOCHS = 10

# Contrastive learning
TEMPERATURE = 0.03

# Pretraining
PRETRAIN_EPOCHS = 3
PRETRAIN_SAVE_PATH = os.path.join(BASE_DIR, "pretrained_clothing_encoder.pth")