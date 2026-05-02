"""
gradio_app.py — SmartWardrobe Demo UI
 
Runs in Colab. Gives a public URL for live demo.
 
Usage:
    !pip install gradio>=4.0 beautifulsoup4 mediapipe
    %run src/gradio_app.py
 
    # OR in a cell:
    import gradio_app
    gradio_app.launch()
"""
 
import sys
import os
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
 
sys.path.insert(0, "/content/SMART-WARDROBE/src")
sys.path.insert(0, "/content/SMART-WARDROBE/demo")  # for scraper + recommender
 
from recommender import SmartWardrobeRecommender
 
 
# ================================================================
# LOAD MODEL ONCE (cached across requests)
# ================================================================
 
_recommender = None
 
def get_recommender():
    global _recommender
    if _recommender is None:
        _recommender = SmartWardrobeRecommender()
    return _recommender
 
 
# ================================================================
# DEMO MODE — curated product list for when scraping fails
# ================================================================
 
DEMO_PRODUCTS = {
    "topwear": [
        {"name": "White Linen Shirt", "price": "₹899",
         "image_url": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=300"},
        {"name": "Black Crop Top", "price": "₹499",
         "image_url": "https://images.unsplash.com/photo-1583744946564-b52ac1c389c8?w=300"},
        {"name": "Floral Blouse", "price": "₹1299",
         "image_url": "https://images.unsplash.com/photo-1571945153237-4929e783af4a?w=300"},
    ],
    "bottomwear": [
        {"name": "High Waist Jeans", "price": "₹1599",
         "image_url": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=300"},
        {"name": "Pleated Skirt", "price": "₹999",
         "image_url": "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=300"},
    ],
}
 
 
# ================================================================
# IMAGE GRID BUILDER
# ================================================================
 
def _fetch_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=8)
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None
 
 
def build_results_grid(results: list[dict], max_cols: int = 5) -> Image.Image:
    """Create a horizontal grid of recommendation images with labels."""
    if not results:
        img = Image.new("RGB", (640, 200), (245, 245, 245))
        d   = ImageDraw.Draw(img)
        d.text((200, 90), "No results found", fill=(100, 100, 100))
        return img
 
    n    = min(len(results), max_cols)
    W, H = 200, 280     # per-card size
    PAD  = 10
 
    canvas = Image.new("RGB", (n * (W + PAD) + PAD, H + PAD*2), (248, 248, 248))
    draw   = ImageDraw.Draw(canvas)
 
    for i, item in enumerate(results[:n]):
        x = PAD + i * (W + PAD)
        y = PAD
 
        # Product image
        img = _fetch_image(item.get("image_url", ""))
        if img:
            img = img.resize((W, W))
            canvas.paste(img, (x, y))
        else:
            draw.rectangle([x, y, x+W, y+W], fill=(220, 220, 220))
            draw.text((x+60, y+90), "No image", fill=(150, 150, 150))
 
        # Rank badge
        draw.ellipse([x+4, y+4, x+30, y+30], fill=(0, 0, 0, 200))
        draw.text((x+10, y+8), f"#{i+1}", fill=(255, 255, 255))
 
        # Similarity score
        sim_str = f"sim={item.get('similarity', 0):.3f}"
        draw.rectangle([x, y+W, x+W, y+W+18], fill=(30, 30, 30))
        draw.text((x+4, y+W+2), sim_str, fill=(200, 200, 200))
 
        # Product name (truncated)
        name = item.get("name", "Product")[:28]
        draw.text((x+4, y+W+22), name, fill=(40, 40, 40))
 
        # Price
        price = item.get("price", "")
        if price:
            draw.text((x+4, y+W+40), price, fill=(80, 80, 80))
 
    return canvas
 
 
# ================================================================
# CORE RECOMMENDATION FUNCTION (called by Gradio)
# ================================================================
 
def run_recommendation(
    user_image,
    website_url: str,
    category: str,
    top_k: int,
    use_measurements: bool,
    height_cm: float,
    bust_cm: float,
    waist_cm: float,
    hip_cm: float,
    shoulder_cm: float,
) -> tuple:
    """
    Main Gradio callback.
    Returns: (results_image, status_text, results_json_str)
    """
    rec = get_recommender()
 
    if not website_url.strip():
        return None, "Please enter a website URL.", "[]"
 
    try:
        # ── Option A: use measurements directly ─────────────────
        if use_measurements:
            results = rec.recommend_from_measurements(
                measurements={
                    "height_cm":   height_cm,
                    "bust_cm":     bust_cm,
                    "waist_cm":    waist_cm,
                    "hip_cm":      hip_cm,
                    "shoulder_cm": shoulder_cm,
                },
                website_url=website_url.strip(),
                category=category.lower(),
                top_k=int(top_k),
            )
 
        # ── Option B: use uploaded photo ─────────────────────────
        elif user_image is not None:
            # Save temp image
            tmp_path = "/tmp/user_photo.jpg"
            if isinstance(user_image, np.ndarray):
                Image.fromarray(user_image).save(tmp_path)
            else:
                user_image.save(tmp_path)
 
            results = rec.recommend(
                user_image_path=tmp_path,
                website_url=website_url.strip(),
                category=category.lower(),
                top_k=int(top_k),
            )
 
        else:
            return None, "Please upload a photo OR enter measurements.", "[]"
 
        if not results:
            return (
                build_results_grid([]),
                "No products found. Try a different URL or category.",
                "[]"
            )
 
        # Build output
        grid   = build_results_grid(results, max_cols=min(5, int(top_k)))
        status = (f"Found {len(results)} recommendations "
                  f"(sim range: {results[-1]['similarity']:.3f}–"
                  f"{results[0]['similarity']:.3f})")
 
        import json
        result_json = json.dumps([
            {k: v for k, v in r.items() if k != "image"}
            for r in results
        ], indent=2)
 
        return grid, status, result_json
 
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return None, f"Error: {e}\n\n{tb}", "[]"
 
 
# ================================================================
# GRADIO INTERFACE
# ================================================================
 
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="SmartWardrobe — Body-Aware Fashion Recommendation",
        theme=gr.themes.Base(
            primary_hue="slate",
            secondary_hue="gray",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("DM Sans"),
        ),
        css="""
        .header { text-align: center; padding: 24px 0 8px; }
        .header h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; }
        .header p  { color: #888; font-size: 0.95rem; margin-top: 6px; }
        .section-label { font-size: 0.75rem; font-weight: 600;
                         text-transform: uppercase; letter-spacing: 0.08em;
                         color: #888; margin-bottom: 6px; }
        .result-box { background: #fafafa; border-radius: 12px; padding: 16px; }
        """
    ) as demo:
 
        gr.HTML("""
        <div class="header">
          <h1>SmartWardrobe</h1>
          <p>Body-aware clothing recommendation · Upload photo or enter measurements · Paste any fashion URL</p>
        </div>
        """)
 
        with gr.Row():
 
            # ── LEFT COLUMN — Inputs ──────────────────────────────
            with gr.Column(scale=1):
 
                gr.HTML('<div class="section-label">Your Body</div>')
                user_image = gr.Image(
                    label="Upload full-body photo",
                    type="numpy",
                    height=300,
                )
 
                use_meas = gr.Checkbox(
                    label="Enter measurements manually instead",
                    value=False
                )
 
                with gr.Group(visible=False) as meas_group:
                    gr.HTML('<div class="section-label">Measurements (cm)</div>')
                    with gr.Row():
                        height_cm   = gr.Slider(140, 200, value=165, step=1, label="Height")
                        bust_cm     = gr.Slider(70,  130, value=88,  step=1, label="Bust")
                    with gr.Row():
                        waist_cm    = gr.Slider(55,  110, value=70,  step=1, label="Waist")
                        hip_cm      = gr.Slider(75,  130, value=96,  step=1, label="Hip")
                    shoulder_cm = gr.Slider(30, 55, value=38, step=1, label="Shoulder width")
 
                use_meas.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=use_meas,
                    outputs=meas_group
                )
 
                gr.HTML('<div class="section-label" style="margin-top:16px">Website & Category</div>')
                website_url = gr.Textbox(
                    label="Fashion website URL",
                    placeholder="https://www.myntra.com/topwear",
                    lines=1,
                )
                category = gr.Dropdown(
                    choices=["topwear", "bottomwear", "dresses", "footwear", "outerwear"],
                    value="topwear",
                    label="Garment category"
                )
                top_k = gr.Slider(3, 20, value=10, step=1, label="Number of recommendations")
 
                recommend_btn = gr.Button(
                    "Get Recommendations",
                    variant="primary",
                    size="lg"
                )
 
            # ── RIGHT COLUMN — Outputs ─────────────────────────────
            with gr.Column(scale=2):
                gr.HTML('<div class="section-label">Recommendations</div>')
                result_image  = gr.Image(label="Top picks", height=320)
                status_text   = gr.Textbox(label="Status", lines=1, interactive=False)
 
                with gr.Accordion("Raw results (JSON)", open=False):
                    result_json = gr.Code(language="json", label="")
 
        # ── Quick-start examples ──────────────────────────────────
        gr.HTML('<div class="section-label" style="margin:20px 0 8px">Try these examples</div>')
        gr.Examples(
            examples=[
                [None, "https://www.myntra.com/topwear",     "topwear",    10, False, 165, 88, 70, 96, 38],
                [None, "https://www.myntra.com/jeans",       "bottomwear", 10, False, 165, 88, 70, 96, 38],
                [None, "https://www.asos.com/women/dresses", "dresses",    10, False, 170, 90, 72, 98, 39],
            ],
            inputs=[user_image, website_url, category, top_k,
                    use_meas, height_cm, bust_cm, waist_cm, hip_cm, shoulder_cm],
            label="",
        )
 
        recommend_btn.click(
            fn=run_recommendation,
            inputs=[user_image, website_url, category, top_k,
                    use_meas, height_cm, bust_cm, waist_cm, hip_cm, shoulder_cm],
            outputs=[result_image, status_text, result_json],
        )
 
    return demo
 
 
def launch(share: bool = True, **kwargs):
    """Launch the Gradio app. share=True gives a public URL."""
    demo = build_ui()
    demo.launch(share=share, **kwargs)
 
 
if __name__ == "__main__":
    launch(share=True)
