# 👗 SMART WARDROBE

A deep learning-based clothing recommendation system that matches outfits to body shape using a joint embedding model inspired by the **ViBE (Visual Body-aware Embedding)** architecture.

---

## 📌 Overview

Smart Wardrobe learns a shared 128-dimensional embedding space where **body measurements** and **clothing images** are brought close together if they are compatible. The system uses a dual-encoder architecture — one for body shape and one for clothing — trained with triplet loss to rank outfit recommendations.

---

## 🧠 Architecture

### ViBEModel (Joint Embedding)
The core model maps both body vectors and clothing images into a shared L2-normalized 128-dim embedding space, where Euclidean distance reflects compatibility.

```bash

Body Measurements (7D)
    └─► BodyEncoder (MLP: 7 → 32 → 64 → 128) ──┐
                                                  ├─► Shared 128-dim Space
Clothing Image (3×224×224)                        │   (Euclidean Distance)
    └─► ClothEncoder (ResNet50 → 512 → 256 → 128)┘
```
## ✨Features

- Body measurement extraction via MediaPipe pose estimation directly from images.
- Bidirectional recommendation: given a body → find best clothing; given clothing → find best body type.
- Triplet loss training with mid-epoch checkpointing every 200 batches.
- Resume training support from checkpoints.
- Visualization of top-5 recommendations with similarity scores.

## 🤝Contributing
Pull requests and suggestions are welcome! Feel free to open an issue for bugs or feature requests.

