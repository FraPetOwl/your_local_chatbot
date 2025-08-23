# This script sets up a FastAPI application that serves as a chatbot API.
# It uses LangChain to handle product data queries from a Chroma vector store.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from embed_products import db
from fastapi.responses import StreamingResponse
import logging


app = FastAPI(debug=True)
logging.basicConfig(level=logging.DEBUG)

# CORS for frontend dev later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str


# Load LLM
llm = Ollama(model="tinyllama", temperature=0.0, num_predict=500)  # Set temperature to 0 for more deterministic responses, num_predict to limit response length

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question

    # Similarity search
    start_embed = time.time()
    raw_docs = db.similarity_search(question, k=15) # Retrieve top 15 documents from the vector store
    print(f"🔍 Embedding + similarity search took {time.time() - start_embed:.2f}s")

    # Deduplicate by title
    unique_products = []
    seen_urls = set()

    for doc in raw_docs:
        url = doc.metadata.get("url")
        if url not in seen_urls:
            unique_products.append(doc)
            seen_urls.add(url)

        if len(unique_products) == 15:
            break

    print(f"Got {len(unique_products)} unique products: {unique_products}")

    start_context = time.time()        
    max_context_length = 2000 
    context = "" # Initialize empty context string

    
    for doc in unique_products:
        next_entry = f"------\n{doc.page_content}" 
        if len(context) + len(next_entry) > max_context_length: 
            break
        context += next_entry

    # Build prompt
    prompt = f"""
        ## Response Guidelines:
        1. Answer questions EXCLUSIVELY using the Chroma database knowledge provided to you
        2. Always cite your sources with BOTH the product title and reference URL
        3. When information is not found in the provided documents, clearly state this limitation
        4. DO NOT rely on general knowledge about AI platforms or related topics when documentation is unavailable
        5. Do not invent new price totals — always list only the product’s actual price as listed in the metadata
        6. Ensure every suggestion is unique and never duplicated
        7. All descriptions must come from the description field of the product metadata
        6. Format all citations at the end of your response as: 
        <<Information retrieved from sources: (URL)

        ## Interaction Approach:
        - Seek clarification when user intent is ambiguous rather than making assumptions
        - Ask targeted follow-up questions to understand user needs and goals better
        - Maintain a helpful, knowledgeable tone focused on practical solutions
        - When appropriate, suggest related features or documentation that might benefit the user

        Remember that your answers should prioritize accuracy over completeness—it's better to acknowledge limitations than provide potentially incorrect information that is not supported by documentation.
            
        ROLE: You are a helpful assistant for a DJ and lighting rental shop called Superior Sounds.
        You can only provide answers based on the following product information:
        {context}

        If you do not find a match for the customer's question in the provided information, respond with 'I don't know' or 'I can't find the user-suggested item'.

        Customer question: {question}
        Answer:
        """.strip()

    
    # Call LLM
    try:
        print(f"📚 Context formatting took {time.time() - start_context:.2f}s")
        llm_start = time.time()

        # Create messages
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"{question}\n\nContext:\n{context}")
        ]

        # 🧠 Streaming response generator
        async def token_stream():
            try:
                stream_start = time.time()
                first_token = True
                async for chunk in llm.astream(messages):
                    if first_token:
                        print(f"🟢 First token after {time.time() - stream_start:.2f}s")
                        first_token = False
                    yield chunk
                print(f"✅ Finished streaming all tokens in {time.time() - stream_start:.2f}s")
            except Exception as e:
                yield f"[ERROR] {str(e)}"

        return StreamingResponse(token_stream(), media_type="text/plain")
        
    except Exception as e:
        return {"error": str(e)}

# Run locally if needed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
