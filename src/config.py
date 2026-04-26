# ================================================================
# CONFIGURATION FILE — SMART WARDROBE
# ================================================================
# This file defines:
# ✔ Dataset paths
# ✔ Model hyperparameters
# ✔ Training configurations (all stages)
# ✔ Saved model paths
#
# Supports:
# Stage 1 → DeepFashion (visual features)
# Stage 2 → Polyvore (compatibility learning)
# Stage 3 → Fashionista (body-aware alignment)
# ================================================================

import os

# ================================================================
# BASE PATHS
# ================================================================

DRIVE_ROOT = "/content/drive/MyDrive/SmartWardrobe"
BASE_DIR   = os.path.join(DRIVE_ROOT, "body_shape_recommendor/data")

# -------------------------
# Fashionista Dataset (Stage 3)
# -------------------------
TRAIN_DIR       = os.path.join(BASE_DIR, "split/train")
VAL_DIR         = os.path.join(BASE_DIR, "split/val")
TEST_DIR        = os.path.join(BASE_DIR, "split/test")
BODY_VECTOR_CSV = os.path.join(BASE_DIR, "body_vectors.csv")

# -------------------------
# DeepFashion Dataset (Stage 1)
# -------------------------
DEEPFASHION_DIR = os.path.join(BASE_DIR, "deepfashion_subset")

# -------------------------
# Polyvore Dataset (Stage 2)
# -------------------------
# Stored as HuggingFace Arrow dataset
# Contains:
#   - image (PIL)
#   - category (string)
#   - item_ID (string → contains outfit grouping)
#
# Items with same set_id belong to same outfit → used for compatibility learning
POLYVORE_ARROW_DIR = "/content/drive/MyDrive/polyvore_dataset/data"

# ================================================================
# MODEL SETTINGS
# ================================================================

IMAGE_SIZE     = 224           # Input size for all models
EMBEDDING_DIM  = 128           # Final embedding dimension
BODY_INPUT_DIM = 7             # Number of body features

# ================================================================
# TRAINING HYPERPARAMETERS
# ================================================================

# ------------------------------------------------
# Stage 1 — DeepFashion (visual representation)
# ------------------------------------------------
PRETRAIN_DEEPFASHION_EPOCHS = 5
PRETRAIN_DEEPFASHION_LR     = 1e-4
PRETRAIN_DEEPFASHION_BATCH  = 64
PRETRAIN_DEEPFASHION_TEMP   = 0.07

# ------------------------------------------------
# Stage 2 — Polyvore (compatibility learning)
# ------------------------------------------------
PRETRAIN_POLYVORE_EPOCHS = 5
PRETRAIN_POLYVORE_LR     = 5e-5      # lower LR → fine-tuning stage
PRETRAIN_POLYVORE_BATCH  = 32
PRETRAIN_POLYVORE_TEMP   = 0.07

# ------------------------------------------------
# Stage 3 — Fashionista (body-aware recommendation)
# ------------------------------------------------
EPOCHS = 20
BATCH_SIZE = 64           # reduce to 32 if GPU OOM
LR = 1e-4
TEMPERATURE = 0.07        # stable value (avoid collapse)

# Early stopping (prevents overfitting)
EARLY_STOP_PATIENCE = 5

# ================================================================
# IMAGE NORMALIZATION (IMPORTANT)
# ================================================================
# Must be SAME across:
# ✔ Training
# ✔ DeepFashion evaluation
# ✔ Polyvore evaluation

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ================================================================
# SAVED MODEL PATHS
# ================================================================

# Stage 1 output
DEEPFASHION_ENCODER_PATH = os.path.join(DRIVE_ROOT, "clothing_encoder_deepfashion.pth")

# Stage 2 output
POLYVORE_ENCODER_PATH = os.path.join(DRIVE_ROOT, "clothing_encoder_polyvore.pth")

# Final trained model (Stage 3)
BEST_MODEL_PATH = os.path.join(DRIVE_ROOT, "best_vibe_model.pth")

# ================================================================
# NOTES
# ================================================================
# ✔ Always ensure transforms use IMAGE_SIZE = 224
# ✔ Keep preprocessing consistent across all stages
# ✔ Use Polyvore + DeepFashion for evaluation (NOT Fashionista)
# ✔ Fashionista is weak supervision → use only qualitatively
# ================================================================