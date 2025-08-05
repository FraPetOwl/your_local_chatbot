from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import json

# Load product data
with open("categorized_products.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def truncate_text(text, max_words=300):
    words = text.split()
    return " ".join(words[:max_words])


# Turn each product into a Document
docs = []
for item in data:
    # Truncate description to first 500 characters to reduce size
    description = item.get("description", "")[:500] if item.get("description") else ""

    content = f"""
Title: {item['title']}
Price: ${item['price']}
Category: {item['category']}
Description: {description}
URL: {item['url']}
    """.strip()

    metadata = {
        "title": item["title"],
        "price": item["price"],
        "url": item["url"],
        "description": description,
        "category": item["category"]
    }

    docs.append(Document(page_content=content, metadata=metadata))

# Set the correct embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma DB
db = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")
db.persist()

print(f"Indexed {len(docs)} products into Chroma DB.")
print("You can now query the vector store using similarity search.")
for i, doc in enumerate(docs[:3], 1):  # Show first 3 for verification
    print(f"\nDocument {i}:\nTitle: {doc.metadata['title']}\nContent: {doc.page_content[:100]}...")  # Show first 100 chars
