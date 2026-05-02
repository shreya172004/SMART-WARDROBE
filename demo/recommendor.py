# recommender_v2.py — FIXED VERSION

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

import subprocess
import json
import os

def run_scraper(url):
    print("Running JS scraper...")

    subprocess.run(
        ["node", "scrapper/scrapper.js", url],
        cwd="/content/SMART-WARDROBE"
    )

    with open("/content/SMART-WARDROBE/products.json") as f:
        products = json.load(f)

    print(f"Scraped {len(products)} products")
    return products

UPPER_BODY_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


class SmartWardrobeRecommender:


    def recommend(self, user_image_path, products, category, top_k=5):
        import torch
        from PIL import Image
        import requests
        from io import BytesIO

        print(f"Using preloaded products: {len(products)}")

        # -------------------------
        # Load user image
        # -------------------------
        user_img = Image.open(user_image_path).convert("RGB")
        user_tensor = self.transform(user_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            user_embedding = self.model(user_tensor)
            user_embedding = user_embedding / user_embedding.norm(dim=1, keepdim=True)

        results = []

        # -------------------------
        # Process products
        # -------------------------
        for p in products:
            try:
                # download product image
                resp = requests.get(p["image_url"], timeout=5)
                img = Image.open(BytesIO(resp.content)).convert("RGB")

                img_tensor = self.transform(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    prod_embedding = self.model(img_tensor)
                    prod_embedding = prod_embedding / prod_embedding.norm(dim=1, keepdim=True)

                # cosine similarity
                similarity = torch.mm(user_embedding, prod_embedding.t()).item()

                # -------------------------
                # OPTIONAL: body prior boost
                # -------------------------
                body_score = 0.5  # default neutral

                final_score = 0.7 * similarity + 0.3 * body_score

                results.append({
                    "name": p.get("name", ""),
                    "image_url": p.get("image_url", ""),
                    "product_url": p.get("product_url", ""),
                    "similarity": similarity,
                    "final_score": final_score
                })

            except Exception as e:
                continue

        # -------------------------
        # Sort results
        # -------------------------
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        # add rank
        for i, r in enumerate(results[:top_k]):
            r["rank"] = i + 1

        print(f"Returning top {top_k} results")

        return results[:top_k]



    def __init__(self,
                 model_path=None,
                 prior_path=None,
                 prior_alpha=0.25,
                 device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.prior_alpha = prior_alpha

        # Load model
        self.model = ViBEModel(
            body_input_dim=config.BODY_INPUT_DIM,
            embedding_dim=config.EMBEDDING_DIM
        ).to(self.device)

        ckpt = model_path or config.BEST_MODEL_PATH
        if os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
            print(f"Model loaded from {ckpt}")
        else:
            print(f"WARNING: checkpoint not found at {ckpt}")

        self.model.eval()

        # Load prior
        self.prior = None
        if prior_path and os.path.exists(prior_path):
            self.prior = BodyShapePrior.load(prior_path)
            print(f"Prior loaded (alpha={prior_alpha})")

    # ─────────────────────────────
    # Encoding
    # ─────────────────────────────

    @torch.no_grad()
    def encode_body(self, body_vec):
        t = torch.tensor(body_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        return F.normalize(self.model.encode_body(t), dim=1).cpu()

    @torch.no_grad()
    def encode_cloth(self, pil_image):
        t = UPPER_BODY_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        return F.normalize(self.model.encode_cloth(t), dim=1).cpu()

    def encode_products(self, products):
        valid, embeddings = [], []

        for p in products:
            img = p.get("image")
            if img is None:
                continue

            try:
                emb = self.encode_cloth(img)
                valid.append(p)
                embeddings.append(emb)
            except:
                continue

        if not embeddings:
            return [], torch.zeros(0, config.EMBEDDING_DIM)

        return valid, torch.cat(embeddings, dim=0)

    # ─────────────────────────────
    # Ranking
    # ─────────────────────────────

    def _score_and_rank(self, body_vec, body_emb, cloth_embs, top_k):

        base_scores = torch.matmul(cloth_embs, body_emb.T).squeeze(1).numpy()

        if self.prior is not None:
            final_scores = self.prior.rerank(
                body_vec=body_vec,
                cloth_embs=cloth_embs.numpy(),
                base_scores=base_scores,
                alpha=self.prior_alpha
            )
        else:
            final_scores = base_scores

        top_idx = np.argsort(-final_scores)[:top_k]
        return top_idx, final_scores

    # ─────────────────────────────
    # MAIN FIX HERE ✅
    # ─────────────────────────────

    def recommend_from_measurements(
        self,
        measurements,
        website_url=None,
        category="topwear",
        top_k=10,
        max_scrape=60,
        products=None   # 🔥 NEW PARAM
    ):

        # Convert measurements → body vector
        h   = measurements.get("height_cm", 165) / 100
        b   = measurements.get("bust_cm", 88) / 100
        w   = measurements.get("waist_cm", 70) / 100
        hip = measurements.get("hip_cm", 96) / 100
        sw  = measurements.get("shoulder_cm", 38) / 100

        body_vec = np.array(
            [h, b, w, hip, sw, sw/(hip+1e-6), (h-hip)*0.5],
            dtype=np.float32
        )

        body_emb = self.encode_body(body_vec)

        # 🔥 KEY FIX: use products if provided
        if products is None:
            print(f"Scraping {website_url}...")
            products = scrape_products(website_url, category, max_scrape)

            if not products:
                print("No products scraped")
                return []

            products = download_product_images(products)

        else:
            print(f"Using preloaded products: {len(products)}")

            # 🔥 Ensure images exist
            products = download_product_images(products)

        valid, cloth_embs = self.encode_products(products)

        if not valid:
            print("No valid products after encoding")
            return []

        top_idx, scores = self._score_and_rank(
            body_vec, body_emb, cloth_embs, top_k
        )

        results = []
        for rank, idx in enumerate(top_idx):
            p = valid[idx].copy()
            p.pop("image", None)

            p["similarity"] = float(scores[idx])
            p["rank"] = rank + 1

            results.append(p)

        return results