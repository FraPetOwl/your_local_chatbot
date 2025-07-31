# This script scrapes the homepage of Superior Sounds Events and saves the text content to a file.
# It uses requests to fetch the page and BeautifulSoup to parse the HTML.

import requests
from bs4 import BeautifulSoup

PAGES = [
    "https://superiorsoundsevents.com/",
    "https://superiorsoundsevents.com/services-1",
    "https://superiorsoundsevents.com/bio",
    "https://superiorsoundsevents.com/testimonials",
    "https://superiorsoundsevents.com/affiliations"
]

def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    clean = "\n".join([line for line in lines if line])

    return f"--- CONTENT FROM: {url} ---\n\n{clean}\n"

if __name__ == "__main__":
    full_text = ""
    for url in PAGES:
        full_text += scrape_page(url) + "\n"

    with open("static_scraped.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    print("✅ Static pages scraped to static_scraped.txt")
