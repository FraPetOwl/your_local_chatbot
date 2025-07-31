# scrape_dynamic.py
# This script scrapes dynamic pages of Superior Sounds Events and saves the text content to files.

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

DYNAMIC_PAGES = {
    "Shop": "https://twice.shop/superiorsounds1/shop"
}

def scrape_dynamic_page(name, url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(5000)  # Wait for JS content to load


        html = page.content()
        browser.close()

        # Save raw HTML (optional)
        with open(f"{name}_raw.html", "w", encoding="utf-8") as f:
            f.write(html)

        # Clean with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()

        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines()]
        clean_lines = [line for line in lines if line]

        with open(f"{name}_clean.txt", "w", encoding="utf-8") as f:
            f.write('\n'.join(clean_lines))

        print(f"✅ Scraped and saved: {name}_clean.txt")

if __name__ == "__main__":
    for name, url in DYNAMIC_PAGES.items():
        scrape_dynamic_page(name, url)
