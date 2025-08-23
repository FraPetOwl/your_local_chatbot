import json

# Load products from JSON
with open("updated_shop_data.json", "r", encoding="utf-8") as f:
    products = json.load(f)

# Define categories
categories = {
    "audio_equipment": ["speaker", "microphone", "mixer", "amplifier"],
    "lighting_equipment": ["light", "led", "laser", "projection"],
    "stage_equipment": ["stage", "truss", "backdrop", "screen"],
    "dj_equipment": ["dj controller", "turntable", "mixer"],
    "video_equipment": ["projector", "screen", "tv"],
    "photo_equipment": ["photo booth"],
    "other": []
}

# Function to categorize a product
def categorize_product(product):
    title = product["title"].lower()
    description = product["description"].lower()

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in title or keyword.lower() in description:
                return category

    return "other"

# Categorize products
for product in products:
    product["category"] = categorize_product(product)

# Save categorized products to JSON
with open("categorized_products.json", "w") as f:
    json.dump(products, f, indent=4)
