from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import json
import shutil
import os

# Start fresh: delete old DB folder
db_dir = "./chroma_db"
if os.path.exists(db_dir):
    shutil.rmtree(db_dir)

# Load product data
with open("categorized_products.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare documents
docs = []
for item in data:
    description = item.get("description", "")
    if not description or description == "N/A":
        continue  # Skip empty descriptions

    content = f"""
    🎛️ From Category: {item['category']} | Item: {item['title']} | Price: ${item['price']}
    {description}
    Available at: {item['url']}
    """.strip()

    metadata = {
        "title": item["title"],
        "price": item["price"],
        "url": item["url"],
        "description": description,
        "category": item["category"]
    }

    docs.append(Document(page_content=content, metadata=metadata))

print(f"✅ Prepared {len(docs)} documents for embedding")

# Set embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and persist DB
db = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")
db.persist()

# Check final DB contents
print(f"✅ Persisted {len(db.get()['documents'])} documents in Chroma DB.")
