"""
recommender_v2.py — SmartWardrobe Recommender with Body-Shape Prior
 
Extends recommender.py by adding optional body-shape prior re-ranking.
Drop-in replacement: same .recommend() and .recommend_from_measurements()
API, just add prior_path argument.
 
Changes from recommender.py:
  - Loads BodyShapePrior if prior_path is given
  - .recommend() and .recommend_from_measurements() blend cosine scores
    with prior scores when prior is available
  - All other logic identical
"""
 
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
 
 
UPPER_BODY_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])
 
 
class BodyExtractor:
    """Same as recommender.py — extracts 7-dim body vector from photo."""
 
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_pose  = mp.solutions.pose
            self.available = True
        except ImportError:
            self.available = False
 
    def _dist(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
 
    def _mid(self, p1, p2):
        return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
 
    def extract(self, image_path: str) -> np.ndarray:
        if not self.available:
            return np.zeros(7, dtype=np.float32)
        import cv2
        img     = cv2.imread(image_path)
        if img is None:
            img = np.array(Image.open(image_path).convert("RGB"))[:,:,::-1]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.mp_pose.Pose(static_image_mode=True,
                                min_detection_confidence=0.4) as pose:
            result = pose.process(img_rgb)
        if not result.pose_landmarks:
            return np.zeros(7, dtype=np.float32)
        lm = result.pose_landmarks.landmark
        def pt(i): return (lm[i].x, lm[i].y)
        try:
            shoulder_w  = self._dist(pt(11), pt(12))
            hip_w       = self._dist(pt(23), pt(24))
            shoulder_mid = self._mid(pt(11), pt(12))
            hip_mid      = self._mid(pt(23), pt(24))
            ankle_mid    = self._mid(pt(27), pt(28))
            height      = self._dist(pt(0), ankle_mid)
            return np.array([
                height, shoulder_w*1.3, hip_w*0.85, hip_w,
                shoulder_w, shoulder_w/(hip_w+1e-6),
                self._dist(shoulder_mid, hip_mid)
            ], dtype=np.float32)
        except Exception:
            return np.zeros(7, dtype=np.float32)
 
 
class SmartWardrobeRecommender:
    """
    Recommender with optional body-shape prior re-ranking.
 
    Args:
        model_path  : path to best_vibe_model.pth
        prior_path  : path to body_prior.pkl (optional but recommended)
        prior_alpha : blend weight for prior (0=no prior, 1=only prior)
                      default 0.25 — subtle adjustment, doesn't override model
    """
 
    def __init__(self,
                 model_path:  str = None,
                 prior_path:  str = None,
                 prior_alpha: float = 0.25,
                 device:      str = None):
 
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device      = torch.device(device)
        self.prior_alpha = prior_alpha
 
        # Load model
        self.model = ViBEModel(
            body_input_dim=config.BODY_INPUT_DIM,
            embedding_dim=config.EMBEDDING_DIM
        ).to(self.device)
        ckpt = model_path or config.BEST_MODEL_PATH
        if os.path.exists(ckpt):
            self.model.load_state_dict(
                torch.load(ckpt, map_location=self.device))
            print(f"  Model loaded from {ckpt}")
        else:
            print(f"  WARNING: checkpoint not found at {ckpt}")
        self.model.eval()
 
        # Load prior (optional)
        self.prior = None
        if prior_path and os.path.exists(prior_path):
            self.prior = BodyShapePrior.load(prior_path)
            print(f"  Prior loaded (alpha={prior_alpha})")
        elif prior_path:
            print(f"  Prior not found at {prior_path}. Run body_prior.py --build first.")
 
        self.body_extractor = BodyExtractor()
 
    # ── Encoding ─────────────────────────────────────────────────────
 
    @torch.no_grad()
    def encode_body(self, body_vec: np.ndarray) -> torch.Tensor:
        t = torch.tensor(body_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        return F.normalize(self.model.encode_body(t), dim=1).cpu()
 
    @torch.no_grad()
    def encode_cloth(self, pil_image: Image.Image) -> torch.Tensor:
        t = UPPER_BODY_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        return F.normalize(self.model.encode_cloth(t), dim=1).cpu()
 
    def encode_products(self, products):
        valid, embeddings = [], []
        for p in products:
            img = p.get("image")
            if img is None: continue
            try:
                emb = self.encode_cloth(img)
                valid.append(p)
                embeddings.append(emb)
            except Exception:
                continue
        if not embeddings:
            return [], torch.zeros(0, config.EMBEDDING_DIM)
        return valid, torch.cat(embeddings, dim=0)
 
    # ── Score blending ───────────────────────────────────────────────
 
    def _score_and_rank(
        self,
        body_vec:   np.ndarray,
        body_emb:   torch.Tensor,
        cloth_embs: torch.Tensor,
        top_k:      int,
    ) -> torch.Tensor:
        """
        Compute final scores, blending model cosine sim with prior if available.
        Returns top_k indices into cloth_embs.
        """
        # Base cosine similarity from model
        base_scores = torch.matmul(cloth_embs, body_emb.T).squeeze(1).numpy()
 
        if self.prior is not None:
            # Blend with body-shape prior
            final_scores = self.prior.rerank(
                body_vec    = body_vec,
                cloth_embs  = cloth_embs.numpy(),
                base_scores = base_scores,
                alpha       = self.prior_alpha,
            )
        else:
            final_scores = base_scores
 
        top_k   = min(top_k, len(final_scores))
        top_idx = np.argsort(-final_scores)[:top_k]
        return top_idx, final_scores
 
    # ── Main recommend ───────────────────────────────────────────────
 
    def recommend(
        self,
        user_image_path: str,
        website_url:     str,
        category:        str = "topwear",
        top_k:           int = 10,
        max_scrape:      int = 60,
    ) -> list[dict]:
 
        print(f"\n{'='*55}\n  SmartWardrobe Recommendation\n{'='*55}")
        print("  Step 1: Extracting body measurements...")
        body_vec = self.body_extractor.extract(user_image_path)
        body_emb = self.encode_body(body_vec)
 
        print(f"  Step 2: Scraping {website_url}...")
        products = scrape_products(website_url, category, max_scrape)
        if not products: return []
 
        print("  Step 3: Downloading product images...")
        products = download_product_images(products, max_workers=8)
        if not products: return []
 
        print("  Step 4: Encoding clothing images...")
        valid, cloth_embs = self.encode_products(products)
        if not valid: return []
 
        print("  Step 5: Ranking" +
              (" with body-shape prior..." if self.prior else "..."))
        top_idx, scores = self._score_and_rank(body_vec, body_emb,
                                                cloth_embs, top_k)
 
        results = []
        for rank, idx in enumerate(top_idx):
            p = valid[idx].copy()
            p.pop("image", None)
            p["similarity"] = round(float(scores[idx]), 4)
            p["rank"]        = rank + 1
            results.append(p)
 
        print(f"\n  Done. Top {len(results)} recommendations ready.")
        return results
 
    def recommend_from_measurements(
        self,
        measurements: dict,
        website_url:  str,
        category:     str = "topwear",
        top_k:        int = 10,
        max_scrape:   int = 60,
    ) -> list[dict]:
 
        h   = measurements.get("height_cm",   165) / 100
        b   = measurements.get("bust_cm",      88)  / 100
        w   = measurements.get("waist_cm",     70)  / 100
        hip = measurements.get("hip_cm",       96)  / 100
        sw  = measurements.get("shoulder_cm",  38)  / 100
        body_vec = np.array(
            [h, b, w, hip, sw, sw/(hip+1e-6), (h-hip)*0.5],
            dtype=np.float32
        )
 
        body_emb = self.encode_body(body_vec)
        products = scrape_products(website_url, category, max_scrape)
        if not products: return []
        products = download_product_images(products)
        valid, cloth_embs = self.encode_products(products)
        if not valid: return []
 
        top_idx, scores = self._score_and_rank(body_vec, body_emb,
                                                cloth_embs, top_k)
        results = []
        for rank, idx in enumerate(top_idx):
            p = valid[idx].copy()
            p.pop("image", None)
            p["similarity"] = round(float(scores[idx]), 4)
            p["rank"]        = rank + 1
            results.append(p)
        return results
 
