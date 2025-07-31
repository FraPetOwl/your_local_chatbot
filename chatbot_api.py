# chatbot_api.py
# This script sets up a FastAPI application to handle questions about products from Superior Sounds Events.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
import logging
import json
import time

# Setup FastAPI and logging
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Init on startup
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_function,
)
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.7}
)
llm = Ollama(
    model="llama2:7b",
    temperature=0.1,
    base_url="http://localhost:11434"
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce",
    return_source_documents=True
)

# Input schema
class QueryInput(BaseModel):
    question: str

# /ask endpoint
@app.post("/ask")
async def ask(input: QueryInput):
    try:
        logger.info(f"Question received: {input.question}")
        start_time = time.time()

        result = qa.invoke({"query": input.question})
        answer = result["result"]
        source_docs = result["source_documents"]

        # Optional: Clean up sources
        cleaned_sources = []
        for doc in source_docs:
            try:
                content = json.loads(doc.page_content)
                parts = []
                for category, items in content.items():
                    parts.append(f"{category}:")
                    for item in items:
                        parts.append(f"- {item.get('name', '')}: ${item.get('price_from', 'N/A')}")
                cleaned_sources.append("\n".join(parts))
            except Exception:
                cleaned_sources.append(doc.page_content)

        return {
            "answer": answer,
            "sources": cleaned_sources,
            "processing_time": f"{time.time() - start_time:.2f}s"
        }

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn chatbot_api:app --reload
