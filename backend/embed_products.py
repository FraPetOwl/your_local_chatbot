# embed_products.py
# Embeds rental product information from categorized_products.json
# into a Chroma vector database for semantic search.

import json
import shutil
import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Synonym mapping for embedding enrichment ---
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

# --- Expand queries / content with synonyms ---
def get_synonyms_for_item(title: str, category: str, description: str) -> list:
    """Return a list of relevant synonyms for a product."""
    text_blob = f"{title} {category} {description}".lower()
    found = set()
    for key, syns in synonyms_map.items():
        if key in text_blob:
            found.update(syns)
    return list(found)

# Get the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- STEP 1: Start fresh by deleting old DB ---
db_dir = os.path.join(script_dir, "chroma_db")
if os.path.exists(db_dir):
    try:
        shutil.rmtree(db_dir)
        print("🗑️ Cleared old database")
    except Exception as e:
        print(f"Error deleting directory: {e}")

# --- STEP 2: Load product data from JSON ---
json_path = os.path.join(script_dir, "categorized_products.json")
try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📁 Loaded {len(data)} items from categorized_products.json")
except FileNotFoundError:
    print(f"❌ {json_path} not found!")
    exit(1)
except json.JSONDecodeError as e:
    print(f"❌ JSON decode error: {e}")
    exit(1)

# --- STEP 3: Prepare documents ---
docs = []
seen_products = set()
processed_count = skipped_duplicates = skipped_empty = 0

for item in data:
    title = item.get("title", "").strip()
    price = item.get("price", "").strip()
    url = item.get("url", "").strip()
    category = item.get("category", "equipment").strip()
    description = item.get("description", "").strip()

    # Skip missing critical info
    if not title or not price or price.lower() in ['n/a', 'na', '']:
        skipped_empty += 1
        continue

    # Deduplicate
    product_key = f"{title.lower()}_{price}"
    if product_key in seen_products:
        skipped_duplicates += 1
        continue
    seen_products.add(product_key)

    # Fallback description
    if not description or description.lower() in ['n/a', 'na']:
        description = f"{category} available for rental"

    # Gather synonyms
    synonyms = get_synonyms_for_item(title, category, description)
    synonym_text = f"\nAlso known as: {', '.join(synonyms)}" if synonyms else ""

    # Prepare content for embedding
    content = f"""
{title} - {category} rental equipment
Price: ${price}
Description: {description}
Category: {category}
Available for rental from Superior Sounds
{synonym_text}
""".strip()

    # Metadata
    metadata = {
        "title": title,
        "price": price,
        "url": url,
        "description": description,
        "category": category,
        "product_type": category  # store full category
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

# --- STEP 5: Create and persist Chroma DB ---
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

# --- STEP 6: Test queries ---
print("\n🔍 Testing database with sample searches...")
test_queries = ["wireless microphone", "LED lighting", "PA system", "fog machine", "DJ controller"]

for query in test_queries:
    results = db.similarity_search(query, k=3)
    print(f"\nQuery: '{query}' -> Found {len(results)} results")
    for i, doc in enumerate(results[:2]):
        title = doc.metadata.get('title', 'Unknown')
        price = doc.metadata.get('price', 'N/A')
        url = doc.metadata.get('url', '')
        print(f"  {i+1}. {title} (${price}/day) Book: {url}")

print(f"\n✅ Final database stats:")
print(f"   Total documents: {len(db.get()['documents'])}")
print(f"   Categories represented: {len(set(doc.metadata.get('category', '') for doc in docs))}")
print(f"   Ready for queries! 🚀")
