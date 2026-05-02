const puppeteer = require("puppeteer");
const fs = require("fs");

async function scrape(url) {
    console.log("Opening:", url);

    const browser = await puppeteer.launch({
        headless: "new",
        args: ["--no-sandbox", "--disable-setuid-sandbox"]
    });

    const page = await browser.newPage();

    await page.setUserAgent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
    );

    try {
        await page.goto(url, {
            waitUntil: "networkidle2",
            timeout: 60000
        });

        // scroll to load dynamic content
        await autoScroll(page);

        const products = await page.evaluate(() => {
            const items = [];

            document.querySelectorAll("a").forEach(el => {
                const img = el.querySelector("img");

                if (img && img.src && el.href) {
                    items.push({
                        name: img.alt || "No name",
                        image_url: img.src,
                        product_url: el.href,
                        price: "N/A"
                    });
                }
            });

            return items;
        });

        await browser.close();

        //  Deduplicate
        const unique = [];
        const seen = new Set();

        for (const p of products) {
            if (!seen.has(p.product_url)) {
                seen.add(p.product_url);
                unique.push(p);
            }
        }

        console.log("Saved products:", products.length);

        return unique.slice(0, 60);
    } catch (err) {
        console.error("Scraping failed:", err.message);
        await browser.close();
        return [];
    }
}

// auto scroll function
async function autoScroll(page) {
    await page.evaluate(async () => {
        await new Promise((resolve) => {
            let totalHeight = 0;
            const distance = 300;

            const timer = setInterval(() => {
                window.scrollBy(0, distance);
                totalHeight += distance;

                if (totalHeight >= document.body.scrollHeight) {
                    clearInterval(timer);
                    resolve();
                }
            }, 300);
        });
    });
}

// CLI entry
const url = process.argv[2];

if (!url) {
    console.log("Usage: node scrapper.js <url>");
    process.exit(1);
}

scrape(url).then(products => {
    fs.writeFileSync(
        "products.json",
        JSON.stringify(products, null, 2)
    );

    console.log("Saved to products.json");
});