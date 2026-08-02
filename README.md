#  Smart Wardrobe

A deep learning-powered clothing recommendation system that matches outfits to body shape using a **joint embedding model** inspired by the **ViBE (Visual Body-aware Embedding)** architecture.

Instead of treating clothing recommendation as a traditional classification problem, Smart Wardrobe learns a **shared embedding space** where compatible body measurements and clothing images are positioned close together. This enables personalized outfit retrieval based on visual compatibility rather than predefined labels.

---

#  Overview

Choosing clothing that complements an individual's body shape is often subjective and requires significant manual effort. Smart Wardrobe addresses this challenge by learning compatibility directly from data.

The project combines:

- Computer Vision
- Deep Learning
- Metric Learning
- Human Pose Estimation

to recommend clothing items that best suit a person's body measurements.

Rather than predicting clothing categories, the model learns **visual-body compatibility** by embedding body measurements and clothing images into a common latent space.

---

#  Architecture

The core model consists of two separate encoders trained jointly using **Triplet Loss**.

- **Body Encoder:** Converts body measurements into a dense embedding.
- **Clothing Encoder:** Converts clothing images into the same embedding space.

Compatible body-clothing pairs produce embeddings that lie close together, while incompatible pairs are pushed farther apart.

```text
Body Measurements (7D)
        │
        ▼
BodyEncoder
(MLP: 7 → 32 → 64 → 128)
        │
        ├──────────────┐
        │              │
        ▼              ▼
      Shared 128-Dimensional
        Embedding Space
   (L2 Normalized Features)
        ▲              ▲
        │              │
        └──────────────┘
        │
Clothing Image (3×224×224)
        │
        ▼
ClothEncoder
(ResNet50 → 512 → 256 → 128)
```

The Euclidean distance between embeddings is used as the compatibility score.

---

#  Features

-  Body measurement extraction using **MediaPipe Pose Estimation**
-  Joint embedding learning for body-clothing compatibility
-  Clothing recommendation based on body measurements
-  Reverse recommendation (find compatible body types for a clothing item)
-  Metric learning using **Triplet Loss**
-  Automatic checkpoint saving every 200 training batches
-  Resume training from saved checkpoints
-  Top-5 recommendation visualization with similarity scores
-  Shared L2-normalized embedding space for efficient retrieval

---

#  Tech Stack

- Python
- PyTorch
- TorchVision
- ResNet50
- MediaPipe
- NumPy
- OpenCV
- Matplotlib

---

#  Model

### Body Encoder

- Input: 7 body measurements
- Architecture:
  - Linear (7 → 32)
  - ReLU
  - Linear (32 → 64)
  - ReLU
  - Linear (64 → 128)
  - L2 Normalization

### Clothing Encoder

- ResNet50 backbone
- Fully Connected Projection:
  - 2048 → 512
  - 512 → 256
  - 256 → 128
- L2 Normalization

---

#  Training

The model is trained using **Triplet Loss**, where:

- **Anchor:** Body embedding
- **Positive:** Compatible clothing
- **Negative:** Incompatible clothing

The objective is to minimize the distance between compatible pairs while maximizing the distance from incompatible pairs.

---

#  Installation

```bash
git clone https://github.com/yourusername/SmartWardrobe.git

cd SmartWardrobe

pip install -r requirements.txt
```

---

# ▶ Usage

Train the model:

```bash
python train.py
```

Generate recommendations:

```bash
python recommend.py
```

---

#  Future Improvements

- Transformer-based clothing encoder
- Multi-garment outfit recommendation
- Seasonal and occasion-aware recommendations
- User preference learning
- Real-time mobile deployment
- Retrieval using Approximate Nearest Neighbors (FAISS)

---

#  Contributing

Contributions, suggestions, and bug reports are welcome. Feel free to fork the repository and submit a pull request.

---

#  Contributors

This project was developed through the **equal collaborative efforts** of:

- **Mansi Gupta**
- **Shreya Mahara**
- **Rudra Chandna**

Each contributor played an equally significant role in the design, development, implementation, experimentation, and evaluation of the Smart Wardrobe system.

---

#  Acknowledgement

This project is inspired by the **ViBE (Visual Body-aware Embedding)** architecture for learning joint body-clothing embeddings. While inspired by ViBE, the implementation has been adapted and extended specifically for the Smart Wardrobe recommendation system.
