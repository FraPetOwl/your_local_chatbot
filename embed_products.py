# embed_products.py
# This script embeds product data from a JSON file and stores it in a local vector database using Chroma.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import os


# 1. Load product data from file
with open("shop_data.json", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = text_splitter.create_documents([text])

# 3. Create local (free) HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store locally using Chroma (no cloud, fast, free)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db"
)
vectorstore.persist()

print("✅ Embedding complete and stored locally with Chroma!")
