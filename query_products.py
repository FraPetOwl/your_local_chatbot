# query_products.py
# This script queries the local vector database for product information using Chroma.

from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load the same embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Chroma vectorstore
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Sample query
query = input("Ask a question about rentals: ")
results = vectorstore.similarity_search(query, k=5)

# Deduplicate results by page_content (you could also parse for URL/title if desired)
seen = set()
unique_results = []
for doc in results:
    if doc.page_content not in seen:
        unique_results.append(doc)
        seen.add(doc.page_content)

# Print unique results
for i, doc in enumerate(unique_results[:3], 1):  # show up to 3 unique results
    print(f"\nResult {i}:\n{doc.page_content}")
