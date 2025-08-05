from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

db = Chroma(persist_directory="./chroma_db", embedding_function=HuggingFaceEmbeddings())
docs = db.similarity_search("light and speaker products", k=5)

for i, doc in enumerate(docs, 1):
    print(f"Doc {i}: {doc.page_content}")
