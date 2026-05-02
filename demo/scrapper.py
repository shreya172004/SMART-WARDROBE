"""
scraper.py — Product scraper for SmartWardrobe demo
 
Scrapes product images + metadata from fashion e-commerce sites.
Works with: Myntra, ASOS, Zara, H&M, Amazon Fashion, and generic sites.
 
Usage:
    products = scrape_products(
        url="https://www.myntra.com/topwear",
        category="topwear",
        max_products=50
    )
    # Returns list of {"image_url", "name", "price", "product_url", "category"}
"""
 
import requests
import time
import random
import re
from urllib.parse import urljoin, urlparse
from io import BytesIO
from PIL import Image
 
# ── Try importing BeautifulSoup ───────────────────────────────────
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("  BeautifulSoup not found. Run: !pip install beautifulsoup4 requests")
 
 
# ================================================================
# HEADERS — rotate to avoid bot detection
# ================================================================
 
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
]
 
 
def _get_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
 
 
# ================================================================
# CATEGORY KEYWORD MAP
# Maps user-selected category to search keywords for filtering
# ================================================================
 
CATEGORY_KEYWORDS = {
    "topwear": [
        "top", "shirt", "blouse", "tee", "t-shirt", "kurta", "crop",
        "tank", "polo", "sweatshirt", "hoodie", "jacket", "cardigan",
        "sweater", "tunic", "camisole"
    ],
    "bottomwear": [
        "pant", "trouser", "jeans", "skirt", "shorts", "legging",
        "palazzo", "culottes", "jogger", "chino"
    ],
    "dresses": [
        "dress", "gown", "maxi", "midi", "mini", "jumpsuit",
        "romper", "co-ord", "suit"
    ],
    "footwear": [
        "shoe", "boot", "sandal", "heel", "flat", "loafer",
        "sneaker", "slipper", "mule", "wedge"
    ],
    "outerwear": [
        "coat", "jacket", "blazer", "trench", "puffer", "windbreaker",
        "overcoat", "shrug"
    ],
}
 
 
def _matches_category(name: str, category: str) -> bool:
    """Check if product name matches the selected category."""
    if not name or category not in CATEGORY_KEYWORDS:
        return True   # if we can't tell, include it
    keywords = CATEGORY_KEYWORDS[category]
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)
 
 
# ================================================================
# IMAGE DOWNLOAD & VALIDATION
# ================================================================
 
def _download_image(url: str, timeout=10) -> Image.Image | None:
    """Download image from URL and validate it's a real clothing image."""
    try:
        resp = requests.get(url, headers=_get_headers(), timeout=timeout)
        if resp.status_code != 200:
            return None
        img = Image.open(BytesIO(resp.content)).convert("RGB")
 
        # Filter out tiny images (icons, logos)
        w, h = img.size
        if w < 100 or h < 100:
            return None
 
        return img
    except Exception:
        return None
 
 
# ================================================================
# GENERIC SCRAPER  (works on most fashion e-commerce sites)
# ================================================================
 
def _scrape_generic(url: str, category: str, max_products: int) -> list[dict]:
    """
    Generic HTML scraper — works on most fashion sites.
    Extracts product cards by looking for common patterns.
    """
    if not BS4_AVAILABLE:
        return []
 
    try:
        resp = requests.get(url, headers=_get_headers(), timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Fetch failed: {e}")
        return []
 
    soup   = BeautifulSoup(resp.text, "html.parser")
    base   = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    products = []
 
    # Common product card selectors across fashion sites
    selectors = [
        "div[class*='product']",
        "div[class*='item']",
        "article[class*='product']",
        "li[class*='product']",
        "div[class*='card']",
    ]
 
    cards = []
    for sel in selectors:
        cards = soup.select(sel)
        if len(cards) >= 3:
            break
 
    if not cards:
        # Fallback: find all images with alt text that looks like clothing
        imgs = soup.find_all("img", alt=True)
        for img in imgs[:max_products * 3]:
            alt = img.get("alt", "")
            if not _matches_category(alt, category):
                continue
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src", "")
            if not src or src.startswith("data:"):
                continue
            src = urljoin(base, src)
            parent = img.find_parent("a")
            product_url = urljoin(base, parent["href"]) if parent and parent.get("href") else url
            products.append({
                "image_url":   src,
                "name":        alt,
                "price":       "",
                "product_url": product_url,
                "category":    category,
            })
            if len(products) >= max_products:
                break
        return products
 
    for card in cards[:max_products * 2]:
        # Extract image
        img_tag = (
            card.find("img") or
            card.find("image")
        )
        if not img_tag:
            continue
 
        img_src = (
            img_tag.get("src") or
            img_tag.get("data-src") or
            img_tag.get("data-lazy-src") or
            img_tag.get("data-original", "")
        )
        if not img_src or img_src.startswith("data:"):
            continue
        img_src = urljoin(base, img_src)
 
        # Extract name
        name = (
            img_tag.get("alt", "") or
            (card.find(["h2", "h3", "h4", "p", "span"],
                        class_=re.compile(r"name|title|product", re.I)) or {}).get_text(strip=True)
        )
 
        if not _matches_category(name, category):
            continue
 
        # Extract price
        price_el = card.find(
            class_=re.compile(r"price|amount|cost", re.I)
        )
        price = price_el.get_text(strip=True) if price_el else ""
 
        # Extract product URL
        link = card.find("a", href=True)
        product_url = urljoin(base, link["href"]) if link else url
 
        products.append({
            "image_url":   img_src,
            "name":        name[:120],
            "price":       price[:30],
            "product_url": product_url,
            "category":    category,
        })
 
        if len(products) >= max_products:
            break
 
        time.sleep(0.1)   # polite delay
 
    return products
 
 
# ================================================================
# SITE-SPECIFIC SCRAPERS
# ================================================================
 
def _scrape_myntra(url: str, category: str, max_products: int) -> list[dict]:
    """Myntra-specific scraper."""
    # Myntra loads products via JS — use their API endpoint
    # The URL format: myntra.com/topwear → we extract the category slug
    parts     = url.rstrip("/").split("/")
    cat_slug  = parts[-1] if parts else "topwear"
    api_url   = (f"https://www.myntra.com/gateway/v2/search/{cat_slug}"
                 f"?p=1&rows={max_products}&o=0&plaEnabled=false")
    try:
        resp = requests.get(api_url, headers=_get_headers(), timeout=15)
        data = resp.json()
        items = data.get("searchData", {}).get("results", {}).get("products", [])
        products = []
        for item in items[:max_products]:
            img = item.get("mediaImages", [{}])[0].get("src", "")
            if img and not img.startswith("http"):
                img = "https://" + img
            name = item.get("productName", item.get("category", ""))
            if not _matches_category(name, category):
                continue
            products.append({
                "image_url":   img,
                "name":        name,
                "price":       str(item.get("price", "")),
                "product_url": f"https://www.myntra.com/{item.get('landingPageUrl', '')}",
                "category":    category,
            })
        return products
    except Exception:
        return _scrape_generic(url, category, max_products)
 
 
def _scrape_asos(url: str, category: str, max_products: int) -> list[dict]:
    """ASOS-specific scraper."""
    # ASOS uses structured data in script tags
    if not BS4_AVAILABLE:
        return _scrape_generic(url, category, max_products)
    try:
        resp = requests.get(url, headers=_get_headers(), timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select("article[class*='product']")
        products = []
        for card in cards[:max_products]:
            img  = card.find("img")
            src  = (img.get("src") or img.get("data-src", "")) if img else ""
            name = (img.get("alt", "") if img else
                    card.find(class_=re.compile("name")).get_text(strip=True)
                    if card.find(class_=re.compile("name")) else "")
            if not _matches_category(name, category):
                continue
            link = card.find("a", href=True)
            href = "https://www.asos.com" + link["href"] if link else url
            products.append({
                "image_url": src, "name": name,
                "price": "", "product_url": href, "category": category
            })
        return products
    except Exception:
        return _scrape_generic(url, category, max_products)
 
 
# ================================================================
# MAIN ENTRY POINT
# ================================================================
 
SITE_SCRAPERS = {
    "myntra.com":  _scrape_myntra,
    "asos.com":    _scrape_asos,
}
 
 
def scrape_products(url: str, category: str = "topwear",
                    max_products: int = 50) -> list[dict]:
    """
    Main scraping function.
 
    Args:
        url          : Full URL of the product listing page
        category     : One of topwear, bottomwear, dresses, footwear, outerwear
        max_products : Maximum number of products to fetch
 
    Returns:
        List of dicts: {image_url, name, price, product_url, category}
    """
    category = category.lower().strip()
    domain   = urlparse(url).netloc.replace("www.", "")
 
    print(f"  Scraping {domain} for {category} (max {max_products})...")
 
    # Use site-specific scraper if available, else generic
    scraper_fn = SITE_SCRAPERS.get(domain, _scrape_generic)
    products   = scraper_fn(url, category, max_products)
 
    # Deduplicate by image URL
    seen  = set()
    clean = []
    for p in products:
        key = p.get("image_url", "")
        if key and key not in seen:
            seen.add(key)
            clean.append(p)
 
    print(f"  Found {len(clean)} products after deduplication")
    return clean
 
 
def download_product_images(products: list[dict],
                             max_workers: int = 4) -> list[dict]:
    """
    Download images for all scraped products.
    Adds 'image' (PIL.Image) key to each product dict.
    Removes products where download fails.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
 
    def _fetch(p):
        img = _download_image(p["image_url"])
        return p, img
 
    valid = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch, p): p for p in products}
        for future in as_completed(futures):
            p, img = future.result()
            if img is not None:
                p["image"] = img
                valid.append(p)
 
    print(f"  Downloaded {len(valid)}/{len(products)} images successfully")
    return valid
 
