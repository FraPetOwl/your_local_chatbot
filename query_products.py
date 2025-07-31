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
query = "Do you offer uplighting for weddings?"
results = vectorstore.similarity_search(query, k=3)

# Print results
for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:\n{doc.page_content}")
