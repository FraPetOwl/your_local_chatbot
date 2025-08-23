from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import asyncio
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

class ChatRequest(BaseModel):
    question: str

# Optimized LLM configuration
llm = Ollama(
    model="qwen2.5:0.5b",  # Much smaller
    temperature=0.1,
    num_predict=500,      # Reduced for faster responses
    num_ctx=2048,         # Reduced context window
    repeat_penalty=1.5,
    top_k=40,
    top_p=0.9
)


SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant for Superior Sounds, a DJ and lighting rental shop.

RULES (follow exactly, no exceptions):
1. Only use the Product Information provided below — copy product names, prices, and descriptions exactly as written.
2. Do not add, change, or guess any details.
3. Include the product title and URL for each recommendation.
4. Keep responses concise and helpful.
5. Format URLs as clickable links.

Product Information:
{context}

"""

def build_optimized_context(docs, max_length=1200):
    """Build context more efficiently with better formatting"""
    context_parts = []
    current_length = 0
    
    for doc in docs:
        # Extract key info efficiently
        title = doc.metadata.get("title", "Unknown")
        price = doc.metadata.get("price", "N/A")
        url = doc.metadata.get("url", "")
        category = doc.metadata.get("category", "equipment")
        description = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content # 
        
        # More natural format for conversation
        product_info = f"- {title} (${price}/day): {description} [Book: {url}]\n"
        
        if current_length + len(product_info) > max_length:
            break
            
        context_parts.append(product_info) 
        current_length += len(product_info)
    
    return "\n".join(context_parts)

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question
    
    try:
        # Similarity search with timing
        start_time = time.time()
        raw_docs = db.similarity_search(question, k=8)  # Reduced from 15
        search_time = time.time() - start_time
        print(f"🔍 Search took {search_time:.2f}s")

        # Deduplicate efficiently
        unique_products = []
        seen_urls = set()
        
        for doc in raw_docs:
            url = doc.metadata.get("url")
            if url and url not in seen_urls:
                unique_products.append(doc)
                seen_urls.add(url)
            if len(unique_products) >= 8:  # Limit to 8 products
                break

        # Response info / Build optimized context
        context_start = time.time()
        context = build_optimized_context(unique_products)
        context_time = time.time() - context_start
        print(f"📚 Context build took {context_time:.2f}s")

        # Create system prompt with context
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        
        # Create single optimized prompt with better formatting
        full_prompt = f"{system_prompt}\n\nCustomer: {question}\n\nRental Assistant:"

        print(f"📝 Prompt length: {len(full_prompt)} characters")
        print(f" Prompt number of words: {len(full_prompt.split())}")
        print(f"Full prompt:\n{full_prompt}\n")

        # Streaming response with better error handling
        async def optimized_stream():
            try:
                llm_start = time.time()
                first_token = True
                token_count = 0

                # Get response (could be AIMessage or str)
                if len(unique_products) == 0:
                    response = "I couldn't find any matching products, but I'm positive Superior Sounds has what you need. Please try a different question, check their website @ https://twice.shop/superiorsounds1/shop?category=KgFiRALzbaIYKlhR0e2D or contact them directly."
                response = await llm.ainvoke(full_prompt)

                # Ensure we have a plain string
                if hasattr(response, "content"):
                    text = response.content
                else:
                    text = str(response)

                # Stream the response word by word for better UX
                words = text.split()
                for i, word in enumerate(words):
                    if first_token:
                        print(f"🟢 First token after {time.time() - llm_start:.2f}s")
                        first_token = False

                    yield word + " "
                    token_count += 1

                    if i % 3 == 0:
                        await asyncio.sleep(0.01)

                total_time = time.time() - llm_start
                print(f"✅ Total LLM time: {total_time:.2f}s, Tokens: {token_count}")

            except Exception as e:
                print(f"❌ LLM Error: {e}")
                yield "I'm having trouble processing your request."

        return StreamingResponse(optimized_stream(), media_type="text/plain")
                
    except Exception as e:
            print(f"❌ General Error: {e}")
            return {"error": "Something went wrong. Please try again."}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "qwen2.5:0.5b"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)