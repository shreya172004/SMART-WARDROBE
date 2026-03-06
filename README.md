# 👗 SMART WARDROBE

A deep learning-based clothing recommendation system that matches outfits to body shape using a joint embedding model inspired by the **ViBE (Visual Body-aware Embedding)** architecture.

---

## 📌 Overview

Smart Wardrobe learns a shared 128-dimensional embedding space where **body measurements** and **clothing images** are brought close together if they are compatible. The system uses a dual-encoder architecture — one for body shape and one for clothing — trained with triplet loss to rank outfit recommendations.

---

## 🧠 Architecture

### ViBEModel (Joint Embedding)
The core model maps both body vectors and clothing images into a shared L2-normalized 128-dim embedding space, where Euclidean distance reflects compatibility.
