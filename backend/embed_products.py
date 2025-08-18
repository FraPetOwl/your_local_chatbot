# embed_products.py
# This script embeds product information from categorized_products.json
# and stores it in a Chroma vector database for semantic search.

import json
import shutil
import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Synonym mapping for product embedding ---
synonyms_map = {
    "microphone": ["mic", "wireless mic", "handheld mic", "vocal microphone"],
    "speaker": ["loudspeaker", "PA system", "sound system", "monitor"],
    "lighting": ["uplight", "spotlight", "stage light", "wash light", "color wash"],
    "dj": ["disc jockey", "turntable", "DJ controller", "mix deck"],
    "mixer": ["audio mixer", "mixing console", "sound board"],
    "haze": ["fog machine", "smoke machine", "atmosphere effect", "special effects"],
    "uplight": ["par can", "stage uplight", "event uplight"],
    "projector": ["video projector", "AV projector", "presentation projector"]
}

# --- Synonym expansion for queries ---
def expand_with_synonyms(query: str) -> str:
    """Expand query terms with domain-specific synonyms for better recall."""
    expansions = {
        'mic': 'microphone',
        'wireless mic': 'wireless microphone',
        'lights': 'lighting',
        'speaker': 'speakers sound audio',
        'dj': 'DJ turntable mixer',
        'wedding': 'event party reception',
        'outdoor': 'outside garden patio'
    }
    search_terms = query.lower()
    for short, expanded in expansions.items():
        if short in search_terms:
            search_terms += f" {expanded}"
    return search_terms

def get_synonyms_for_item(title, category, description):
    """Find relevant synonyms for a product based on title/category/description."""
    text_blob = f"{title} {category} {description}".lower()
    found = set()
    for key, syns in synonyms_map.items():
        if key in text_blob:
            found.update(syns)
    return list(found)

# --- STEP 1: Start fresh by deleting old DB ---
db_dir = "./chroma_db"
if os.path.exists(db_dir):
    try:
        shutil.rmtree(db_dir)
        print("🗑️ Cleared old database")
    except Exception as e:
        print(f"Error deleting directory: {e}")

# --- STEP 2: Load product data from JSON ---
try:
    with open("categorized_products.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📁 Loaded {len(data)} items from categorized_products.json")
except FileNotFoundError:
    print("❌ categorized_products.json not found!")
    exit(1)

# --- STEP 3: Prepare documents for embedding ---
docs = []
seen_products = set()
processed_count = 0
skipped_duplicates = 0
skipped_empty = 0

for item in data:
    title = item.get("title", "").strip()
    price = item.get("price", "").strip()
    url = item.get("url", "").strip()
    category = item.get("category", "equipment").strip()
    description = item.get("description", "").strip()
    
    # Skip if missing critical info
    if not title or not price or price.lower() in ['n/a', 'na', '']:
        skipped_empty += 1
        continue

    # Deduplicate based on title+price
    product_key = f"{title.lower()}_{price}"
    if product_key in seen_products:
        skipped_duplicates += 1
        continue
    seen_products.add(product_key)

    # Fallback description
    if not description or description.lower() in ['n/a', 'na', '']:
        description = f"{category} available for rental"

    # Gather synonyms for embedding
    synonyms = get_synonyms_for_item(title, category, description)
    synonym_text = ""
    if synonyms:
        synonym_text = "\nAlso known as: " + ", ".join(synonyms)

    # Content for embedding
    content = f"""
{title} - {category} rental equipment
Price: ${price}
Description: {description}
Category: {category}
Available for rental from Superior Sounds
{synonym_text}
""".strip()

    # Metadata for retrieval filtering
    metadata = {
        "title": title,
        "price": price,
        "url": url,
        "description": description,
        "category": category,
        "product_type": category.split()[0] if category else "equipment"
    }

    docs.append(Document(page_content=content, metadata=metadata))
    processed_count += 1

print(f"✅ Prepared {processed_count} documents")
print(f"⏭️ Skipped {skipped_duplicates} duplicates")
print(f"⏭️ Skipped {skipped_empty} items with missing data")

if not docs:
    print("❌ No valid documents to embed!")
    exit(1)

# --- STEP 4: Initialize embedding model ---
print("🧠 Initializing embedding model...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# --- STEP 5: Create and persist DB ---
print("💾 Creating vector database...")
start_time = time.time()

db = Chroma.from_documents(
    docs,
    embedding,
    persist_directory=db_dir,
    collection_metadata={"hnsw:space": "cosine"}
)
db.persist()

embed_time = time.time() - start_time
print(f"✅ Database created in {embed_time:.2f} seconds")

# --- STEP 6: Verify with test queries ---
print("\n🔍 Testing database with sample searches...")
test_queries = [
    "wireless microphone",
    "LED lighting",
    "PA system",
    "fog machine",
    "DJ controller"
]

for query in test_queries:
    results = db.similarity_search(query, k=3)
    print(f"\nQuery: '{query}' -> Found {len(results)} results")
    for i, doc in enumerate(results[:2]):
        title = doc.metadata.get('title', 'Unknown')
        price = doc.metadata.get('price', 'N/A')
        print(f"  {i+1}. {title} (${price}/day)")

print(f"\n✅ Final database stats:")
print(f"   Total documents: {len(db.get()['documents'])}")
print(f"   Categories represented: {len(set(doc.metadata.get('category', '') for doc in docs))}")
print(f"   Ready for queries! 🚀")
