import os
 
# ================================================================
# BASE PATHS
# ================================================================
BASE_DIR = "/content/drive/MyDrive/SmartWardrobe/body_shape_recommendor/data"
DRIVE_ROOT = "/content/drive/MyDrive/SmartWardrobe"
 
# Fashionista splits
TRAIN_DIR       = os.path.join(BASE_DIR, "split/train")
VAL_DIR         = os.path.join(BASE_DIR, "split/val")
TEST_DIR        = os.path.join(BASE_DIR, "split/test")
BODY_VECTOR_CSV = os.path.join(BASE_DIR, "body_vectors.csv")
 
# DeepFashion (Stage 1 pretraining — visual garment features)
DEEPFASHION_DIR = os.path.join(BASE_DIR, "deepfashion_subset")
 
# ----------------------------------------------------------------
# Polyvore  (Stage 2 pretraining — outfit compatibility)
#
# Format: HuggingFace Arrow shards (loaded via datasets.load_from_disk)
#
# Drive structure:
#   polyvore_dataset/
#     data/
#       dataset_info.json
#       state.json
#       data-00000-of-00006.arrow
#       data-00001-of-00006.arrow
#       ...
#       data-00005-of-00006.arrow
#
# POLYVORE_ARROW_DIR  →  the "data" folder containing the .arrow shards.
# load_from_disk() reads dataset_info.json + state.json automatically.
#
# Dataset schema (Marqo/polyvore, 94,100 rows):
#   image    : PIL Image
#   category : str   e.g. "Day Dresses", "Boots", "Handbags"
#   text     : str   e.g. "tibi knit long sleeve dress"
#   item_ID  : str   e.g. "100002074_1"
#                         ─────────── ─
#                         set_id      item_index
#   Outfit grouping: items with the same set_id prefix (before "_")
#   are from the same outfit → treated as compatible pairs.
# ----------------------------------------------------------------
POLYVORE_ARROW_DIR = "/content/drive/MyDrive/polyvore_dataset/data"
 
# ================================================================
# MODEL SETTINGS
# ================================================================
IMAGE_SIZE     = 224
EMBEDDING_DIM  = 128
BODY_INPUT_DIM = 7
 
# ================================================================
# TRAINING HYPERPARAMETERS
# ================================================================
 
# Stage 1 — DeepFashion visual pretraining
PRETRAIN_DEEPFASHION_EPOCHS = 5
PRETRAIN_DEEPFASHION_LR     = 1e-4
PRETRAIN_DEEPFASHION_BATCH  = 64
PRETRAIN_DEEPFASHION_TEMP   = 0.07
 
# Stage 2 — Polyvore compatibility pretraining
PRETRAIN_POLYVORE_EPOCHS = 5
PRETRAIN_POLYVORE_LR     = 5e-5     # lower: fine-tuning on stage-1 weights
PRETRAIN_POLYVORE_BATCH  = 32
PRETRAIN_POLYVORE_TEMP   = 0.07
 
# Stage 3 — Fashionista body alignment
BATCH_SIZE   = 32
LR           = 1e-4
EPOCHS       = 10
TEMPERATURE  = 0.07   # FIXED: was 0.03 → caused loss collapse
 
# Early stopping
EARLY_STOP_PATIENCE = 3
 
# ================================================================
# SAVED CHECKPOINT PATHS
# ================================================================
DEEPFASHION_ENCODER_PATH = os.path.join(DRIVE_ROOT, "clothing_encoder_deepfashion.pth")
POLYVORE_ENCODER_PATH    = os.path.join(DRIVE_ROOT, "clothing_encoder_polyvore.pth")
BEST_MODEL_PATH          = os.path.join(DRIVE_ROOT, "best_vibe_model.pth")
 
