import os

# =========================
# PATHS
# =========================

BASE_DIR = "/content/drive/MyDrive/SmartWardrobe/body_shape_recommender/data"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

BODY_VECTOR_CSV = os.path.join(BASE_DIR, "body_vectors.csv")

# =========================
# TRAINING PARAMS
# =========================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EMBEDDING_DIM = 128
LR = 1e-4
EPOCHS = 10