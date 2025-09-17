# chatbotV3_optimized.py - this one i want to deploy to show as MVP
# Optimized, streaming-safe version
# Start with: uvicorn chatbotV3_optimized:app --reload --port 8000

import time
import asyncio
import os
import logging
from typing import List, Dict, Optional
import concurrent.futures
import functools

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


try:
    from llama_cpp import Llama
except Exception:
    Llama = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbotV3_optimized")

app = FastAPI(title="Superior Sounds Chatbot", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-2wfu932fg-frapetowls-projects.vercel.app"],  # Your specific frontend
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Superior Sounds Chatbot API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.options("/chat")
async def chat_options():
    return {"message": "CORS preflight OK"}

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=800)

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# SMART CANNED RESPONSES, this is for common questions we can answer without LLM
CANNED_RESPONSES = {
    'wireless mic': {
        'intro': "Here are our wireless microphone options:",
        'products': ['Sennheiser XSW 1-825 Wireless Microphone', 'Sennheiser XSW 1-835 Dual', 'Shure SLXD24D/SM58 Dual']
    },
    'dj speaker': {
        'intro': "Here are our DJ speaker options:",
        'products': ['QSC K10', 'QSC K12', 'QSC-KW122']
    },
    'moving head': {
        'intro': "Here are our moving head lighting options:",
        'products': ['Chauvet Intimidator Spot 350', 'Chauvet Intimidator Hybrid 140SR']
    },
    'haze machine': {
        'intro': "Here are our haze and fog machines:",
        'products': ['Marq 700 Haze Machine']
    },
    'karaoke': {
        'intro': "Perfect for karaoke! Here's what you need:",
        'products': ['Sennheiser XSW 1-825 Wireless Microphone', 'QSC K10', 'Allen & Heath Zed-10FX']
    },
    'retirement party': {
        'intro': "For a large retirement party, consider these options:",
        'products': ['QSC K12', 'Chauvet Intimidator Spot 350', 'Allen & Heath Zed-10FX', 'Photo Booth']
    }
}

POLICY_RESPONSES = {
    "delivery": "We offer delivery! Please contact us for details and pricing.",
    "warranty": "All rentals are professionally maintained. Please ask staff about warranty policies.",
    "sell": "Superior Sounds only offers rentals and service for party needs - we do not sell any products.",
    "t-shirt": "We do not offer clothing or merchandise at this time."
}

# DATABASE SETUP
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

try:
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    logger.info("✅ Database loaded")
except Exception as e:
    logger.error(f"❌ Database error: {e}")
    raise

# LLM SETUP
llm = None
if os.getenv("LLM_ENABLED", "1") != "0" and Llama is not None:
    try:
        llm = Llama.from_pretrained(
            repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
            filename="smollm2-1.7b-instruct-q4_k_m.gguf",
            n_ctx=1024,
            n_threads=2,
            verbose=False
        )
        logger.info("✅ LLM loaded")
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        llm = None

# HELPERS
def format_response(intro: str, products: List[dict]) -> str:
    response = intro + "\n\n"
    for p in products[:6]:
        title = p.get('title', 'Unknown')
        price = p.get('price', 'N/A')
        url = p.get('url', 'https://twice.shop/superiorsounds1/shop')
        desc = p.get('description', 'rental equipment')
        
        response += (
            f"## • **{title}**"
            f" *(${price}/day)* " 
            f"📝*Details:* {desc} "
            f"[Book below👇]{url}\n"
        )
    return response.strip()

def get_canned_response(query: str) -> Optional[tuple]:
    query_lower = query.lower()
    for key, data in CANNED_RESPONSES.items():
        if key in query_lower:
            return data['intro'], data['products']
    for key, reply in POLICY_RESPONSES.items():
        if key in query_lower:
            return reply, None
    return None

async def search_products(query: str, k: int = 6) -> List[dict]:
    loop = asyncio.get_running_loop()
    try:
        docs = await loop.run_in_executor(
            EXECUTOR,
            functools.partial(db.similarity_search, query, k)
        )
        products = []
        seen = set()
        for doc in docs:
            title = doc.metadata.get("title", "").strip()
            if title and title not in seen:
                seen.add(title)
                products.append({
                    "title": title,
                    "price": doc.metadata.get("price", "N/A"),
                    "url": doc.metadata.get("url", ""),
                    "description": doc.metadata.get("description", "Professional rental equipment")
                })
        return products
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

async def get_products_by_names(product_names: List[str]) -> List[dict]:
    loop = asyncio.get_running_loop()
    products = []
    for name in product_names:
        docs = await loop.run_in_executor(
            EXECUTOR,
            functools.partial(db.similarity_search, name, 2)
        )
        if docs:
            best = docs[0]
            products.append({
                "title": best.metadata.get("title", name),
                "price": best.metadata.get("price", "N/A"),
                "url": best.metadata.get("url", ""),
                "description": best.metadata.get("description", "Professional rental equipment")
            })
        else:
            products.append({
                "title": name,
                "price": "N/A",
                "url": "",
                "description": "Not found in database"
            })
    return products

def create_llm_prompt(products: List[dict], query: str) -> str:
    product_list = "\n".join([
        f"- {p['title']}: ${p['price']}/day, {p['description']}, URL: {p['url']}"
        for p in products
    ])
    return f"""You are Superior Sounds rental assistant.

User asked: "{query}"

Available products:
{product_list}

Respond with a short intro sentence or two that addresses the question, includes the type of products found(sound, lighting, stage, dj, or general party equipment), and what they can offer for the user. Do not list specific products."""

def _llm_stream(prompt: str, query: str):
    if not llm:
        logger.warning("⚠️ LLM requested but not available")
        return
    try:
        logger.info("🔄 Starting LLM generation...")
        start_time = time.time() 
        stream = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=110,
            stream=True
        )
        token_count = 0
        for chunk in stream:
            if "choices" in chunk:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    token_count += 1
                    yield delta
        logger.info(f"✅ LLM generated {token_count} tokens and took {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return

# GENERATOR WRAPPER
def string_stream(lines: List[str]):
    for line in lines:
        yield line + "\n"

async def combined_stream(prompt: str, query: str, products: List[dict]):
    llm_response = ""
    # LLM
    if llm:
        for token in _llm_stream(prompt, query):
            llm_response += token
            yield token
        print(f"LLM: {llm_response.strip()}") # prints llm response to console
        
    # Always yield products
    yield "\n\n"
    for p in products[:6]:
        title = p.get('title', 'Unknown')
        price = p.get('price', 'N/A')
        url = p.get('url', '')
        desc = p.get('description', 'Professional rental equipment')
        
        yield f"## • **{title}** *(${price}/day)* 📝*Details:* {desc} [Book below👇]{url}\n\n"

# MAIN ENDPOINT
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        query = body.get("question", "")
        style = body.get("style", "concise")

        canned = get_canned_response(query)
        if canned:
            logger.info(f"🤖 CANNED RESPONSE used for query: '{query}'")
            intro, product_names = canned
            if product_names:
                products = await get_products_by_names(product_names)
                return StreamingResponse(string_stream([format_response(intro, products)]),
                                         media_type="text/plain")
            else:
                # Policy response
                return StreamingResponse(string_stream([intro]), media_type="text/plain")

        logger.info(f"🧠 LLM RESPONSE used for query: '{query}'")
        products = await search_products(query)
        prompt = create_llm_prompt(products, query)
        return StreamingResponse(combined_stream(prompt, query, products), media_type="text/plain")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbotV3_optimized:app", host="0.0.0.0", port=8000, reload=True)

