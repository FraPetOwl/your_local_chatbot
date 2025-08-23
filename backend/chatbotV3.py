# chatbotV3.py
# This script implements an optimized chatbot API for Superior Sounds Events using FastAPI and LangChain.
# It includes enhanced query expansion, intent detection, and improved response generation.
# It also features better error handling, logging, and streaming responses in order to provide help users to find the right rental equipment.

import uvicorn
import json
import re
import time
import asyncio
import os
import logging
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbotV3")

app = FastAPI(debug=True, title="Superior Sounds Chatbot API", version="4.0.0")

# Resolve important paths relative to this file for portability (local/dev/deploy)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    style: str = Field(default="concise", description="Response style preference")

# Initialize components
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

try:
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

    logger.info("✅ Database loaded successfully")
    import sys, os
    logger.info(f"Python exec: {sys.executable}")
    logger.info(f"Using Chroma persist_directory: {CHROMA_DIR}, count: {getattr(db._collection, 'count', lambda: 'unknown')()}")



except Exception as e:
    logger.error(f"❌ Failed to load database: {e}")
    raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Optional LLM toggle via env var (set LLM_ENABLED=0 to force fallback-only)
LLM_ENABLED = os.getenv("LLM_ENABLED", "1").strip() not in ("0", "false", "False")
llm_model_filename = os.getenv("LLM_GGUF_FILENAME", "Llama-3.2-3B-Instruct-Q6_K_L.gguf")
llm_repo_id = os.getenv("LLM_REPO_ID", "bartowski/Llama-3.2-3B-Instruct-GGUF")

llm_threads = int(os.getenv("LLM_THREADS", "8"))
llm_batch = int(os.getenv("LLM_BATCH", "512"))
llm_ctx = int(os.getenv("LLM_CTX", "2048"))

llm = None
if LLM_ENABLED and Llama is not None:
    try:
        llm = Llama.from_pretrained(
            repo_id=llm_repo_id,
            filename=llm_model_filename,
            n_ctx=llm_ctx,
            n_threads=llm_threads,
            n_batch=llm_batch,
            verbose=False
        )
        logger.info(f"LLM loaded: repo_id={llm_repo_id}, filename={llm_model_filename}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        llm = None

# Enhanced synonym expansion with more specific terms
SYNONYM_EXPANSIONS = {
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
    """Enhanced query expansion."""
    expanded_terms = [query.lower()]
    query_lower = query.lower()
    
    for term, expansion in SYNONYM_EXPANSIONS.items():
        if term in query_lower:
            expanded_terms.append(expansion)
    
    return ' '.join(expanded_terms)

def detect_query_intent(query: str) -> str:
    """Detect what type of response the user wants."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['karaoke', 'party', 'wedding', 'event', 'corporate']):
        return 'event_package'
    elif any(word in query_lower for word in ['recommend', 'suggest', 'need', 'what would', 'complete']):
        return 'recommendation'
    elif any(word in query_lower for word in ['affordable', 'cheap', 'budget', 'least expensive']):
        return 'budget_focused'
    elif any(word in query_lower for word in ['do you have', 'which', 'what', 'show me']):
        return 'product_search'
    else:
        return 'general'

# Enhanced context building with better deduplication and metadata to improve response quality

def build_enhanced_context(docs: List[Document], max_items: int = 6) -> List[dict]:
    """Build enhanced context with better product data."""
    items = []
    seen_titles = set()
    
    for doc in docs:
        
        price = doc.metadata.get("price", "").strip()
        url = doc.metadata.get("url", "").strip()
        category = doc.metadata.get("category", "").strip()
        description = doc.metadata.get("description") or doc.page_content or "" # description assigned from metadata or page content
        
        title = doc.metadata.get("title", "").strip() # Avoid empty or duplicate titles
        if not title or (title.lower(), price) in seen_titles: # if title is empty or already seen, skip
            continue
        seen_titles.add(title.lower()) # track seen titles in lowercase for case-insensitive deduplication

        # Keep more description for better responses
        if len(description) > 1000:
            description = description[:997] + "..."
            
        items.append({
            "title": title,
            "price": price,
            "description": description,
            "url": url,
            "category": category
        })
        
        if len(items) >= max_items:
            break
    
    return items

def create_intent_based_prompt(context: List[dict], intent: str) -> str:
    """Create prompts based on query intent for better responses."""
    
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
    
    # Format context as bullet list instead of JSON
    product_list_str = "\n".join(
    f"[PRODUCT]\nTitle: {p['title']}\nPrice: ${p['price']}/day\nCategory: {p['category']}\nURL: {p['url']}\n[/PRODUCT]"
    for p in context
)
    
    return f"""{base_rules}

{instruction}

EXAMPLE RESPONSES:
- "Yes! We have wireless microphones like the Sennheiser XSW 1-825 ($75/day). Book: [URL]"
- "For karaoke nights, you'll need microphones and a mixer. We have the Sennheiser wireless mic ($75/day) and Yamaha mixer ($150/day). Book: [URL1] [URL2]"

PRODUCTS:
{product_list_str}"""

def create_smart_fallback(products: List[dict], query: str, intent: str) -> str:
    """Create intelligent fallback responses based on intent."""
    if not products:
        return "I couldn't find that specific item in our current inventory. Please try a different search or contact us for more options."
    
    query_lower = query.lower()
    
    if intent == 'event_package' or 'karaoke' in query_lower:
        # For events, suggest multiple items
        audio_items = [p for p in products if any(term in p.get('title', '').lower() or p.get('category', '').lower() 
                                                 for term in ['microphone', 'speaker', 'mixer', 'audio'])]
        
        if 'karaoke' in query_lower and audio_items:
            lines = ["For karaoke night, here's what I recommend:"]
            for item in audio_items[:3]:
                title = item.get('title', 'Unknown')
                price = item.get('price', 'Call')
                url = item.get('url', '')
                lines.append(f"• {title} - ${price}/day. Book: {url}")
            return "\n".join(lines)
    
    elif intent == 'budget_focused':
        # Sort by price if possible
        try:
            price_sorted = sorted(products, key=lambda x: float(x.get('price', '999')) if x.get('price', '').replace('.','').isdigit() else 999)
            products = price_sorted[:3]
        except:
            pass
        
        lines = ["Here are our most affordable options:"]
        for item in products[:3]:
            title = item.get('title', 'Unknown')
            price = item.get('price', 'Call')
            url = item.get('url', '')
            lines.append(f"• {title} - ${price}/day. Book: {url}")
        return "\n".join(lines)
    
    # Default product listing
    lines = ["Here are matching items from our rental inventory:"]
    for item in products[:4]:
        title = item.get('title', 'Unknown')
        price = item.get('price', 'Call')
        url = item.get('url', '')
        category = item.get('category', '')
        
        if price and price.lower() not in ['call', 'n/a']:
            price_text = f"${price}/day"
        else:
            price_text = "Call for pricing"
            
        lines.append(f"• {title} ({category}) - {price_text}")
        if url:
            lines.append(f"  Book: {url}")
    
    return "\n".join(lines)

def validate_and_enhance_response(response: str, products: List[dict], query: str) -> Optional[str]:
    """Validate and enhance LLM response."""
    if not response or len(response.strip()) < 15:
        return None
    
    response_lower = response.lower()
    
    # Check for banned content
    banned = ['yuan', 'rmb', 'amazon', 'ebay', 'purchase online', 'buy it', 'for sale']
    if any(term in response_lower for term in banned):
        logger.warning("Response contained banned terms")
        return None
    
    # Check for wrong business model
    if any(term in response_lower for term in ['buy', 'purchase', 'sell']) and 'rent' not in response_lower:
        logger.warning("Response used purchase language")
        return None
    
     # ✅ Ensure all mentioned product names are in retrieved product set
    product_titles = [p['title'].lower() for p in products]
    words_in_response = response_lower.split()
    for word in words_in_response:
        if len(word) > 3 and word not in " ".join(product_titles):
            # Possible hallucination — replace with fallback
            logger.warning(f"Possible hallucinated product: {word}")
            return None
    
    return response.strip()

async def search_products(question: str, k: int = 12) -> List[Document]:
    """Optimized product search."""
    try:
        expanded = expand_query(question)
        logger.debug(f"Expanded query: {expanded[:100]}...")
        
        # Use similarity search with timeout protection
        start_time = time.time()
        results = db.similarity_search(expanded, k=k)
        search_time = time.time() - start_time
        
        logger.info(f"Found {len(results)} products in {search_time:.2f}s")
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    start_time = time.time()
    
    logger.info(f"📝 Question: {question}")
    
    try:
        # Detect intent
        intent = detect_query_intent(question)
        logger.info(f"🎯 Intent: {intent}")
        
        # Search for products
        search_start = time.time()
        raw_products = await search_products(question, k=12)
        search_time = time.time() - search_start
        logger.info(f"🔍 Search: {search_time:.2f}s")
        
        # Build enhanced context
        products = build_enhanced_context(raw_products, max_items=6)
        
        
        async def stream_response():
            try:
                if not products:
                    response = "I couldn't find that item in our current inventory. Please try a different search or contact us for more options."
                    for word in response.split():
                        yield word + " "
                        await asyncio.sleep(0.01)
                    return

                # Create intent-based prompt
                system_prompt = create_intent_based_prompt(products, intent)

                if llm is None:
                    logger.info("🧠 Using fallback response (LLM disabled)...")
                    fallback = create_smart_fallback(products, question, intent)
                    for word in fallback.split():
                        yield word + " "
                        await asyncio.sleep(0.01)
                else:
                    logger.info("🧠 Querying local GGUF model (streaming)...")
                    llm_start = time.time()

                    try:
                        # Stream tokens directly from llama_cpp
                        for chunk in llm.create_chat_completion(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": question}
                            ],
                            temperature=0.2,
                            max_tokens=1024,
                            stream=True
                        ):
                            if "choices" in chunk:
                                delta = chunk["choices"][0]["delta"]
                                if "content" in delta:
                                    token = delta["content"]
                                    yield token
                                    await asyncio.sleep(0)  # let event loop breathe

                        llm_time = time.time() - llm_start
                        logger.info(f"🧠 LLM streamed in: {llm_time:.2f}s")

                    except Exception as e:
                        logger.error(f"LLM error: {e}")
                        fallback = create_smart_fallback(products, question, intent)
                        for word in fallback.split():
                            yield word + " "
                            await asyncio.sleep(0.01)

            except Exception:
                logger.exception("Streaming error")
                error_msg = "Sorry, I'm having technical difficulties. Please try again."
                for word in error_msg.split():
                    yield word + " "
                    await asyncio.sleep(0.01)
        
        # Return the stream to client
        return StreamingResponse(stream_response(), media_type="text/plain")
        
    except Exception:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        total_time = time.time() - start_time
        logger.info(f"⏱️ Total: {total_time:.2f}s")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "4.0.0-optimized",
        "model": "Llama-3.2-3B-Instruct-Q6_K_L.gguf",
        "database_loaded": db._collection.count() if hasattr(db._collection, "count") else "unknown"
    }

@app.get("/debug-chroma-info")
def debug_chroma_info():
    import os
    coll = getattr(db, "_collection", None)
    info = {
        "persist_dir": os.path.abspath("./chroma_db"),
        "collection_type": str(type(coll)),
        "has_count": hasattr(coll, "count")
    }
    try:
        info["count"] = coll.count()
    except Exception as e:
        info["count_error"] = str(e)
    try:
        rows = coll.get(limit=1)
        info["sample_keys"] = list(rows.keys())
        info["sample_meta"] = rows.get("metadatas", rows.get("metadata", []))[:1]
    except Exception as e:
        info["sample_error"] = str(e)
    return info


if __name__ == "__main__":
    uvicorn.run("chatbotV3:app", host="127.0.0.1", port=8000, reload=True)