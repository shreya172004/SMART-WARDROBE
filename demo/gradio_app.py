# gradio_app.py — FINAL CLEAN VERSION

import sys
import os
import gradio as gr

sys.path.insert(0, "/content/SMART-WARDROBE/src")
sys.path.insert(0, "/content/SMART-WARDROBE/demo")

from recommendor import SmartWardrobeRecommender


# =========================================================
# LOAD MODEL ONCE
# =========================================================

rec = SmartWardrobeRecommender(
    model_path="/content/drive/MyDrive/SmartWardrobe/best_vibe_model.pth",
    prior_path="/content/drive/MyDrive/SmartWardrobe/body_prior.pkl",
    prior_alpha=0.25
)


# =========================================================
# CORE FUNCTIONS
# =========================================================

def recommend_measurements(height, bust, waist, hip, shoulder, url):

    if not url:
        return []

    results = rec.recommend_from_measurements(
        measurements={
            "height_cm": height,
            "bust_cm": bust,
            "waist_cm": waist,
            "hip_cm": hip,
            "shoulder_cm": shoulder,
        },
        website_url=url,
        category="topwear",
        top_k=5
    )

    # Return images + captions
    return [
        (r["image_url"], f"{r['name']} | {r.get('similarity', 0):.3f}")
        for r in results
    ]


def recommend_image(image, url):

    if image is None or not url:
        return []

    results = rec.recommend(
        user_image_path=image,
        website_url=url,
        category="topwear",
        top_k=5
    )

    return [
        (r["image_url"], f"{r['name']} | {r.get('similarity', 0):.3f}")
        for r in results
    ]


# =========================================================
# UI
# =========================================================

with gr.Blocks() as app:

    gr.Markdown("# 👗 SmartWardrobe AI")
    gr.Markdown("Enter measurements OR upload image + website URL")

    # -------------------------------
    # TAB 1 — MEASUREMENTS
    # -------------------------------
    with gr.Tab("Measurements"):

        h = gr.Number(label="Height (cm)", value=165)
        b = gr.Number(label="Bust (cm)", value=88)
        w = gr.Number(label="Waist (cm)", value=70)
        hip = gr.Number(label="Hip (cm)", value=96)
        s = gr.Number(label="Shoulder (cm)", value=38)

        url = gr.Textbox(label="Website URL")

        btn = gr.Button("Recommend")

        output = gr.Gallery(label="Results", columns=5)

        btn.click(
            recommend_measurements,
            inputs=[h, b, w, hip, s, url],
            outputs=output
        )

    # -------------------------------
    # TAB 2 — IMAGE
    # -------------------------------
    with gr.Tab("Upload Image"):

        img = gr.Image(type="filepath", label="Upload Photo")

        url2 = gr.Textbox(label="Website URL")

        btn2 = gr.Button("Recommend")

        output2 = gr.Gallery(label="Results", columns=5)

        btn2.click(
            recommend_image,
            inputs=[img, url2],
            outputs=output2
        )


# =========================================================
# LAUNCH
# =========================================================

def launch():
    app.launch(share=True)


if __name__ == "__main__":
    launch()