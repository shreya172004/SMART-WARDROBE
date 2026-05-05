# ================================================================
# SmartWardrobe Recommender (FIXED WITH PROPER NORMALIZATION)
# ================================================================

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/content/SMART-WARDROBE/src")
sys.path.insert(0, "/content/SMART-WARDROBE/demo")

import config
from model import ViBEModel
from scrapper import scrape_products, download_product_images
from body_prior import BodyShapePrior

# =========================
# YOLO BODY VECTOR
# =========================
from ultralytics import YOLO
import cv2

# load once
yolo_pose_model = YOLO("yolov8n-pose.pt")

def extract_body_vector_yolo(image_path):
    """
    Extract 7D body measurements from pose keypoints using YOLO.
    Returns: [height, bust, waist, hip, shoulder_width, ratio, torso]
    """
    img = cv2.imread(image_path)

    if img is None:
        return np.zeros(7, dtype=np.float32)

    results = yolo_pose_model(img, verbose=False)[0]

    if results.keypoints is None or len(results.keypoints) == 0:
        return np.zeros(7, dtype=np.float32)

    kpts = results.keypoints.xy[0].cpu().numpy()

    # COCO indices
    nose = kpts[0]
    left_shoulder = kpts[5]
    right_shoulder = kpts[6]
    left_hip = kpts[11]
    right_hip = kpts[12]
    left_ankle = kpts[15]
    right_ankle = kpts[16]

    def dist(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def mid(p1, p2):
        return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

    shoulder_mid = mid(left_shoulder, right_shoulder)
    hip_mid = mid(left_hip, right_hip)
    ankle_mid = mid(left_ankle, right_ankle)

    height = dist(nose, ankle_mid)
    shoulder_w = dist(left_shoulder, right_shoulder)
    hip_w = dist(left_hip, right_hip)

    bust = shoulder_w * 1.3
    waist = hip_w * 0.85
    ratio = shoulder_w / (hip_w + 1e-6)
    torso = dist(shoulder_mid, hip_mid)

    return np.array([
        height, bust, waist,
        hip_w, shoulder_w,
        ratio, torso
    ], dtype=np.float32)


# ================================================================
# 🔥 BODY VECTOR NORMALIZATION (NEW!)
# ================================================================

def normalize_body_vector(body_vec, method='height_relative'):
    """
    Normalize body measurements to remove scale dependency.
    
    Args:
        body_vec: [height, bust, waist, hip, shoulder_width, ratio, torso]
        method: 'height_relative' (divide by height) or 'standard' (z-score)
    
    Returns:
        Normalized body vector (same shape)
    """
    if method == 'height_relative':
        # Simple but effective: make all measurements relative to height
        height = body_vec[0] + 1e-6  # avoid division by zero
        normalized = body_vec.copy()
        normalized = normalized / height
        return normalized
    
    elif method == 'standard':
        # Standard normalization using population statistics
        # These are typical values - ideally you'd compute from your training set
        
        # Mean and std for [height, bust, waist, hip, shoulder, ratio, torso]
        # Based on typical human body proportions (in meters)
        BODY_MEAN = np.array([1.65, 0.88, 0.70, 0.96, 0.38, 0.40, 0.50], dtype=np.float32)
        BODY_STD = np.array([0.10, 0.10, 0.08, 0.10, 0.05, 0.08, 0.10], dtype=np.float32)
        
        normalized = (body_vec - BODY_MEAN) / (BODY_STD + 1e-6)
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ================================================================
# IMAGE TRANSFORM (FOR CLOTH ENCODER ONLY!)
# ================================================================
# This normalization is ONLY for images going into ResNet
# It has NOTHING to do with body vector normalization

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std =[0.229, 0.224, 0.225]   # ImageNet std
    )
])


# ================================================================
# CLASS
# ================================================================
class SmartWardrobeRecommender:

    def __init__(self, model_path=None, prior_path=None, prior_alpha=0.25, 
                 device=None, body_norm_method='height_relative'):
        """
        Args:
            model_path: Path to model checkpoint
            prior_path: Path to body shape prior
            prior_alpha: Weight for prior (not currently used)
            device: 'cuda' or 'cpu'
            body_norm_method: 'height_relative' or 'standard'
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.prior_alpha = prior_alpha
        self.body_norm_method = body_norm_method

        # Load model
        self.model = ViBEModel(
            body_input_dim=config.BODY_INPUT_DIM,
            embedding_dim=config.EMBEDDING_DIM
        ).to(self.device)

        ckpt = model_path or config.BEST_MODEL_PATH

        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            print(f"✅ Model loaded from {ckpt}")
        else:
            print(f"⚠️  WARNING: checkpoint not found at {ckpt}")

        self.model.eval()

        # Load prior
        self.prior = None
        if prior_path and os.path.exists(prior_path):
            self.prior = BodyShapePrior.load(prior_path)
            print(f"✅ Prior loaded (alpha={prior_alpha})")


    # ============================================================
    # ENCODERS
    # ============================================================

    @torch.no_grad()
    def encode_body(self, body_vec):
        """
        Encode body measurements into embedding space.
        
        Args:
            body_vec: Raw body measurements [7]
        
        Returns:
            Normalized embedding [1, embedding_dim]
        """
        # 🔥 NORMALIZE BODY VECTOR FIRST!
        normalized_vec = normalize_body_vector(body_vec, method=self.body_norm_method)
        
        # Convert to tensor
        t = torch.tensor(normalized_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Encode and normalize embedding
        return F.normalize(self.model.encode_body(t), dim=1).cpu()

    @torch.no_grad()
    def encode_cloth(self, pil_image):
        """
        Encode clothing image into embedding space.
        
        Args:
            pil_image: PIL Image object
        
        Returns:
            Normalized embedding [1, embedding_dim]
        """
        # IMAGE_TRANSFORM includes ImageNet normalization
        t = IMAGE_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        
        # Encode and normalize embedding
        return F.normalize(self.model.encode_cloth(t), dim=1).cpu()


    def encode_products(self, products):
        """
        Batch encode product images.
        
        Args:
            products: List of product dicts with 'image' key
        
        Returns:
            valid_products: List of successfully encoded products
            embeddings: Tensor of shape [N, embedding_dim]
        """
        valid, embeddings = [], []

        for p in products:
            img = p.get("image")

            if img is None:
                continue

            try:
                emb = self.encode_cloth(img)
                valid.append(p)
                embeddings.append(emb)
            except Exception as e:
                print(f"⚠️  Encoding failed: {e}")
                continue

        if not embeddings:
            return [], torch.zeros(0, config.EMBEDDING_DIM)

        return valid, torch.cat(embeddings, dim=0)


    # ============================================================
    # RANKING
    # ============================================================

    def _rank(self, sims, top_k):
        """Return indices of top-k highest similarities."""
        return np.argsort(-sims)[:top_k]


    # ============================================================
    # IMAGE MODE (WITH PROPER BODY NORMALIZATION)
    # ============================================================

    def recommend(self, user_image_path, products, category, top_k=5):
        """
        Recommend products based on user image.
        
        Pipeline:
        1. Try YOLO pose extraction → body encoder
        2. If pose fails → use visual similarity (cloth encoder)
        
        Args:
            user_image_path: Path to user's photo
            products: List of product dicts
            category: Product category (not currently used)
            top_k: Number of recommendations
        
        Returns:
            List of top-k products with similarity scores
        """
        print(f"📦 Using {len(products)} preloaded products")

        # 🔥 YOLO BODY PIPELINE WITH PROPER NORMALIZATION
        try:
            # Extract raw body measurements
            body_vec = extract_body_vector_yolo(user_image_path)

            if np.all(body_vec == 0):
                print("⚠️  YOLO pose detection failed")
                print("↳ Falling back to visual similarity")

                user_img = Image.open(user_image_path).convert("RGB")
                user_emb = self.encode_cloth(user_img)

            else:
                print(f"✅ Body vector extracted: {body_vec}")
                print(f"↳ Normalizing with method: {self.body_norm_method}")
                
                # encode_body() now handles normalization internally
                user_emb = self.encode_body(body_vec)
                
                print("✅ Body embedding created")

        except Exception as e:
            print(f"⚠️  Body pipeline error: {e}")
            print("↳ Falling back to visual similarity")

            user_img = Image.open(user_image_path).convert("RGB")
            user_emb = self.encode_cloth(user_img)

        # Download product images
        products = download_product_images(products)

        # Encode products
        valid, cloth_embs = self.encode_products(products)

        if not valid:
            print("❌ No valid products after encoding")
            return []

        # Compute similarities
        sims = torch.matmul(cloth_embs, user_emb.T).squeeze(1).numpy()

        # Rank
        top_idx = self._rank(sims, top_k)

        # Format results
        results = []
        for rank, idx in enumerate(top_idx):
            p = valid[idx].copy()
            p.pop("image", None)  # Remove PIL object

            p["similarity"] = float(sims[idx])
            p["rank"] = rank + 1

            results.append(p)

        return results


    # ============================================================
    # MEASUREMENT MODE (WITH PROPER NORMALIZATION)
    # ============================================================

    def recommend_from_measurements(
        self,
        measurements,
        website_url=None,
        category="topwear",
        top_k=10,
        max_scrape=60,
        products=None
    ):
        """
        Recommend products based on manual measurements.
        
        Args:
            measurements: Dict with keys height_cm, bust_cm, waist_cm, hip_cm, shoulder_cm
            website_url: URL to scrape (if products not provided)
            category: Product category
            top_k: Number of recommendations
            max_scrape: Max products to scrape
            products: Preloaded products (optional)
        
        Returns:
            List of top-k products with similarity scores
        """
        # Convert cm to meters
        h   = measurements.get("height_cm", 165) / 100
        b   = measurements.get("bust_cm", 88) / 100
        w   = measurements.get("waist_cm", 70) / 100
        hip = measurements.get("hip_cm", 96) / 100
        sw  = measurements.get("shoulder_cm", 38) / 100

        # Construct body vector
        body_vec = np.array(
            [h, b, w, hip, sw, sw/(hip+1e-6), (h-hip)*0.5],
            dtype=np.float32
        )

        print(f"📏 Input measurements: {body_vec}")

        # 🔥 Encode with normalization (handled inside encode_body)
        body_emb = self.encode_body(body_vec)

        # Get products
        if products is None:
            print(f"🌐 Scraping {website_url}...")
            products = scrape_products(website_url, category, max_scrape)
        else:
            print(f"📦 Using {len(products)} preloaded products")

        # Download product images
        products = download_product_images(products)

        # Encode products
        valid, cloth_embs = self.encode_products(products)

        if not valid:
            print("❌ No valid products after encoding")
            return []

        # Compute similarities
        sims = torch.matmul(cloth_embs, body_emb.T).squeeze(1).numpy()

        # Rank
        top_idx = self._rank(sims, top_k)

        # Format results
        results = []
        for rank, idx in enumerate(top_idx):
            p = valid[idx].copy()
            p.pop("image", None)

            p["similarity"] = float(sims[idx])
            p["rank"] = rank + 1

            results.append(p)

        return results


# ================================================================
# USAGE EXAMPLE
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SmartWardrobe Recommender - Fixed Version")
    print("=" * 60)
    print()
    print("✅ Separate normalization pipelines:")
    print("   • Image pipeline: ImageNet normalization (for ResNet)")
    print("   • Body pipeline: Height-relative normalization (for MLP)")
    print()
    print("=" * 60)
    
    # Example initialization
    recommender = SmartWardrobeRecommender(
        body_norm_method='height_relative'  # or 'standard'
    )
    
    print("\n✅ Recommender initialized with proper normalization")