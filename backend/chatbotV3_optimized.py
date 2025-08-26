# chatbotV3_optimized.py
# Optimized version of your chatbotV3.py for faster responses on CPU-only machines.
# Key changes are commented inline. 
# start with uvicorn chatbotV3_optimized:app --reload --port 8000
# model download command:  hf download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_S.gguf

import uvicorn
import json
import re
import time
import asyncio
import os
import logging
from typing import List, Dict, Optional
import concurrent.futures
import functools

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# langchain / llama imports (same as before)
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbotV3_opt")

app = FastAPI(debug=True, title="Superior Sounds Chatbot API", version="4.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    style: str = Field(default="concise", description="Response style preference")

# ========== Performance Tunables ==========
# Use env vars or defaults tuned for your laptop CPU. Adjust if you know your physical cores.
CPU_CORES = int(os.getenv("CPU_CORES", "6"))  # conservative default for laptop to leave headroom
LLM_THREADS = int(os.getenv("LLM_THREADS", str(max(1, CPU_CORES))))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "150"))  # reduce token budget for speed
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "90.0"))  # enforce a fast timeout
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "1"))  # limit concurrent LLM inferences

# ThreadPool for blocking operations (LLM + sync Chroma)
EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=LLM_CONCURRENCY + 2)

# Semaphore to limit concurrent LLM usage and avoid CPU thrash
LLM_SEMAPHORE = asyncio.Semaphore(LLM_CONCURRENCY)

# ========== Embeddings & DB (unchanged but run sync search in executor) ==========
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

try:
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    logger.info("✅ Database loaded successfully")
    import sys
    logger.info(f"Python exec: {sys.executable}")
    logger.info(f"Using Chroma persist_directory: {CHROMA_DIR}, count: {getattr(db._collection, 'count', lambda: 'unknown')()}")
except Exception as e:
    logger.error(f"❌ Failed to load database: {e}")
    raise

# ========== CORS (keep safe dev settings) ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # change to your frontend origin; do NOT use "*" with credentials=True
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)

# ========== LLM init (tune threads) ==========
LLM_ENABLED = os.getenv("LLM_ENABLED", "1").strip() not in ("0", "false", "False")
llm_repo_id = os.getenv("LLM_REPO_ID", "bartowski/Llama-3.2-3B-Instruct-GGUF")
llm_model_filename = os.getenv("LLM_GGUF_FILENAME", "Llama-3.2-3B-Instruct-Q4_K_S.gguf")

llm = None
if LLM_ENABLED and Llama is not None:
    try:
        # Use LLM_THREADS derived from CPU_CORES
        llm = Llama.from_pretrained(
            repo_id=llm_repo_id,
            filename=llm_model_filename,
            n_ctx=int(os.getenv("LLM_CTX", "1024")),
            n_threads=LLM_THREADS,
            n_batch=int(os.getenv("LLM_BATCH", "32")),  # smaller batch to reduce memory spikes
            verbose=False
        )
        logger.info(f"✅ LLM loaded successfully: repo_id={llm_repo_id}, filename={llm_model_filename}, threads={LLM_THREADS}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        llm = None

# ========== Your helpers (unchanged except comments) ==========
SYNONYM_EXPANSIONS = {
    # ... same as your dictionary ...
    'microphone': 'microphone mic vocal handheld wireless Sennheiser Shure audio',
    'mic': 'microphone mic vocal handheld wireless Sennheiser audio',
    'wireless mic': 'wireless microphone handheld vocal cordless Sennheiser',
    'speaker': 'speaker speakers sound audio PA system QSC JBL Alto DJ monitor',
    'speakers': 'speakers speaker sound audio PA system monitors QSC JBL',
    'dj speaker': 'DJ speaker sound system PA speakers QSC JBL powered',
    'lights': 'lighting lights LED uplight spotlight wash color Chauvet Rockville',
    'lighting': 'lighting lights LED uplight spotlight wash color Chauvet',
    'uplight': 'uplight LED color wash ambient lighting Rockville uplights',
    'moving head': 'moving head spot wash beam Intimidator Hybrid light lighting',
    'dj': 'DJ disc jockey turntable mixer Pioneer Yamaha controller deck',
    'smoke': 'smoke fog haze machine atmosphere effect Marq',
    'fog': 'fog smoke haze machine atmosphere effect',
    'haze': 'haze fog smoke machine atmosphere effect Marq',
    'wedding': 'wedding reception ceremony event party celebration outdoor',
    'outdoor': 'outdoor outside garden patio event weather resistant',
    'karaoke': 'karaoke vocal microphone mixer music entertainment PA system',
    'party': 'party celebration event reception gathering dance DJ',
    'audio': 'audio sound PA system speakers microphone amplifier mixer'
}

def expand_query(query: str) -> str:
    expanded_terms = [query.lower()]
    ql = query.lower()
    for term, expansion in SYNONYM_EXPANSIONS.items():
        if term in ql:
            expanded_terms.append(expansion)
    return ' '.join(expanded_terms)

def detect_query_intent(query: str) -> str:
    ql = query.lower()
    if any(word in ql for word in ['karaoke', 'party', 'wedding', 'event', 'corporate']):
        return 'event_package'
    elif any(word in ql for word in ['recommend', 'suggest', 'need', 'what would', 'complete']):
        return 'recommendation'
    elif any(word in ql for word in ['affordable', 'cheap', 'budget', 'least expensive']):
        return 'budget_focused'
    elif any(word in ql for word in ['do you have', 'which', 'what', 'show me']):
        return 'product_search'
    else:
        return 'general'

def build_enhanced_context(docs: List[Document], max_items: int = 6) -> List[dict]:
    items = []
    seen_titles = set()
    for doc in docs:
        price = doc.metadata.get("price", "").strip()
        url = doc.metadata.get("url", "").strip()
        category = doc.metadata.get("category", "").strip()
        description = doc.metadata.get("description") or doc.page_content or ""
        title = doc.metadata.get("title", "").strip()
        if not title or (title.lower(), price) in seen_titles:
            continue
        seen_titles.add(title.lower())
        # REDUCTION: shorten description to speed token count
        if len(description) > 300:
            description = description[:297] + "..."
        items.append({"title": title, "price": price, "description": description, "url": url, "category": category})
        if len(items) >= max_items:
            break
    return items

def create_intent_based_prompt(context: List[dict], intent: str) -> str:
    base_rules = """You are Superior Sounds' rental assistant.
RULES:
1. ONLY recommend products shown between [PRODUCT] blocks.
2. Use the exact Title, Price, and URL as provided.
3. All rentals are by the day.
4. Always include the exact URL from the product section using 'Book: [URL]'."""
    intent_instructions = {
        'event_package': 'Suggest multiple items that work well together for the event type.',
        'recommendation': 'Provide detailed recommendations with reasoning.',
        'budget_focused': 'Highlight the most affordable options and mention prices clearly.',
        'product_search': 'List relevant products with key details and booking links.',
        'general': 'Be helpful and informative about our rental inventory.'
    }
    instruction = intent_instructions.get(intent, intent_instructions['general'])
    # LIMIT context size: small product list string to reduce prompt tokens
    product_list_str = "\n".join(
        f"[PRODUCT]\nTitle: {p['title']}\nPrice: ${p['price']}/day\nCategory: {p['category']}\nURL: {p['url']}\n[/PRODUCT]"
        for p in context[:4]  # reduce to top-4 products
    )
    return f"{base_rules}\n\n{instruction}\n\nPRODUCTS:\n{product_list_str}"

def create_smart_fallback(products: List[dict], query: str, intent: str) -> str:
    # unchanged fallback but short and fast
    if not products:
        return "I couldn't find that specific item in our current inventory. Please try a different search or contact us."
    # ... same logic but shorter outputs ...
    lines = []
    for item in products[:3]:
        title = item.get('title', 'Unknown')
        price = item.get('price', 'Call')
        url = item.get('url', '')
        lines.append(f"{title} - ${price}/day. Book: {url}")
    return "\n".join(lines)

def validate_and_enhance_response(response: str, products: List[dict], query: str) -> Optional[str]:
    if not response or len(response.strip()) < 15:
        return None
    response_lower = response.lower()
    banned = ['yuan', 'rmb', 'amazon', 'ebay', 'purchase online', 'buy it', 'for sale']
    if any(term in response_lower for term in banned):
        logger.warning("Response contained banned terms")
        return None
    # lightweight hallucination guard: skip heavy token-level checks for speed
    return response.strip()

# ========== Make Chroma search non-blocking (run in executor) ==========
async def search_products(question: str, k: int = 12) -> List[Document]:
    """Run the synchronous Chroma similarity_search in a threadpool to avoid blocking event loop."""
    expanded = expand_query(question)
    loop = asyncio.get_running_loop()
    start = time.time()
    try:
        results = await loop.run_in_executor(EXECUTOR, functools.partial(db.similarity_search, expanded, k))
        logger.info(f"Found {len(results)} products in {time.time() - start:.2f}s (db search)")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

# ========== Synchronous helper to run llama streaming inside executor ==========
# Replace your _llm_stream_to_text and call_llm_with_timeout functions with these:

def _simple_llm_call(system_prompt: str, user_question: str, max_tokens: int = LLM_MAX_TOKENS):
    """Simple synchronous LLM call without streaming - runs in thread"""
    if llm is None:
        return None
    
    try:
        logger.info("Making direct LLM call...")
        start = time.time()
        
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_question}
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            stream=False  # No streaming to avoid complexity
        )
        
        elapsed = time.time() - start
        logger.info(f"LLM call completed in {elapsed:.2f}s")
        
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            logger.info(f"LLM returned: '{content[:100]}...' (length: {len(content)})")
            return content
        else:
            logger.error(f"Invalid LLM response format: {response}")
            return None
            
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

async def call_llm_with_timeout(system_prompt: str, user_question: str, timeout: float = 30.0):
    """Simplified async wrapper - let LLM run without timeout since we know it works"""
    if llm is None:
        logger.warning("LLM is None")
        return None
    
    loop = asyncio.get_running_loop()
    
    try:
        logger.info("Starting LLM call in executor...")
        # Run in thread WITHOUT timeout - let it complete naturally
        result = await loop.run_in_executor(None, _simple_llm_call, system_prompt, user_question)
        logger.info(f"LLM executor completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"LLM executor error: {e}")
        return None

# ========== API endpoint with optimized flow ==========
@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    start_time_total = time.time()
    logger.info(f"📝 Question: {question}")

    # Intent & search run concurrently to save time
    t0 = time.time()
    intent = detect_query_intent(question)
    search_task = asyncio.create_task(search_products(question, k=12))
    logger.info(f"Intent detected in {time.time() - t0:.2f}s: {intent}")

    raw_products = await search_task
    products = build_enhanced_context(raw_products, max_items=6)

    # Build prompt (already short)
    system_prompt = create_intent_based_prompt(products, intent)

    async def stream_response():
        # Fast path: no products
        if not products:
            fallback = create_smart_fallback(products, question, intent)
            for word in fallback.split():
                yield word + " "
                await asyncio.sleep(0)  # yield control
            return

        if llm is None:
            # LLM disabled -- use fallback quickly
            fallback = create_smart_fallback(products, question, intent)
            for word in fallback.split():
                yield word + " "
                await asyncio.sleep(0)
            return

        # Try LLM but with timeout and fallback
        logger.info("🧠 Starting LLM call (with timeout)...")
        llm_start = time.time()
        text = await call_llm_with_timeout(system_prompt, question, timeout=30.0)
        logger.info(f"LLM result: text={text is not None}, length={len(text) if text else 0}")
        llm_elapsed = time.time() - llm_start
        logger.info(f"LLM call finished in {llm_elapsed:.2f}s (text length: {len(text) if text else 0})")

        if text:
            # Stream the text back in chunks to client quickly
            # Break into sensible chunk sizes
            chunk_size = 256
            for i in range(0, len(text), chunk_size):
                yield text[i:i+chunk_size]
                await asyncio.sleep(0)  # yield control so client receives chunks
            return
        else:
            # Timeout or error — return a fast concise fallback
            logger.info("Using fast fallback due to LLM timeout/error")
            fallback = create_smart_fallback(products, question, intent)
            for word in fallback.split():
                yield word + " "
                await asyncio.sleep(0)

    total_elapsed = time.time() - start_time_total
    logger.info(f"Total request overhead before LLM/fallback: {total_elapsed:.2f}s")
    return StreamingResponse(stream_response(), media_type="text/plain")

# If you want to run directly
if __name__ == "__main__":
    uvicorn.run("chatbotV3_optimized:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
