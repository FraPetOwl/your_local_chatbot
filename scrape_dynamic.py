# scrape_dynamic.py
# This script scrapes dynamic pages of Superior Sounds Events and saves the text content to files.

import json
import logging
import time
from playwright.sync_api import sync_playwright
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='🧲 - %(message)s')

SHOP_URL = "https://twice.shop/superiorsounds1/shop"

class ProductScraper:
    def __init__(self, browser):
        self.browser = browser
        self.page = browser.new_page()

    def get_product_links(self):
        try:
            self.page.goto(SHOP_URL, timeout=10000)
            self.page.wait_for_timeout(3000)  # Allow JS to load
            product_links = self.page.eval_on_selector_all(
                'a[href^="/superiorsounds1/product/"]',
                "elements => elements.map(el => el.href)"
            )
            return list(set(product_links))
        except Exception as e:
            logging.error(f"Error getting product links: {e}")
            return []

    def scrape_product_page(self, url):
        try:
            logging.info(f"Scraping: {url}")
            self.page.goto(url, timeout=20000)
            self.page.wait_for_timeout(3000)

            # Title
            title = self.page.query_selector("h1")
            if title:
                title = title.text_content().strip()
            else:
                title = "N/A"

            # Price
            price_element = self.page.query_selector("text=Book from")
            if price_element:
                price_text = price_element.text_content().strip()
                price_match = re.search(r'Book from (\d+)', price_text)
                if price_match:
                    price = price_match.group(1)
                else:
                    price = "N/A"
            else:
                price = "N/A"

            # Description
            description_elements = self.page.query_selector_all("span.MuiTypography-root.MuiTypography-body1.css-1klwp6n, div.MuiTypography-root.MuiTypography-body1.css-1j5t6yf > p.MuiTypography-root.MuiTypography-body1.css-1klwp6n")
            if description_elements:
                description = " ".join([element.text_content().strip() for element in description_elements])
            else:
                description = "N/A"


            # Calendar
            '''available_dates = set()
            try:
                self.page.wait_for_selector("button.react-calendar__tile:not([disabled])", timeout=5000)
                buttons = self.page.query_selector_all("button.react-calendar__tile:not([disabled])")
                for btn in buttons:
                    text = btn.query_selector("span.react-calendar__tile-day")
                    if text:
                        available_dates.add(text.text_content().strip())
            except Exception as e:
                logging.error(f"Error scraping calendar: {e}")'''


            return {
                "url": url,
                "title": title,
                "price": price,
                "description": description,
                #"available_dates": list(available_dates)
            }
        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
            return {
                "url": url,
                "title": "ERROR",
                "price": "N/A",
                "description": "N/A",
                "available_dates": []
            }

    def scrape_all_products(self):
        product_links = self.get_product_links()
        logging.info(f"Found {len(product_links)} product links")

        results = []
        for i, link in enumerate(product_links, 1):
            try:
                logging.info(f"Scraping ({i}/{len(product_links)}): {link}")
                data = self.scrape_product_page(link)
                results.append(data)
            except Exception as e:
                logging.error(f"Error scraping {link}: {e}")
            time.sleep(0.5)

        self.browser.close()
        return results

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        scraper = ProductScraper(browser)
        data = scraper.scrape_all_products()
        with open("shop_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info("Saved structured data to shop_data.json")

if __name__ == "__main__":
    main()
