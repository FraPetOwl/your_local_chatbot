# chatbotV3.py
# Superior Sounds — FastAPI chatbot API
# - Loads Chroma DB created by embed_products.py (persisted in ./chroma_db)
# - Provides /chat streaming endpoint that:
#   * expands the user's query with synonyms
#   * searches the vectorstore (with scores when available)
#   * builds a compact JSON context for the LLM
#   * asks a local Ollama LLM to answer using ONLY the provided JSON context
#   * sanitizes the LLM response and falls back to a deterministic product listing if the LLM refuses/fails
#
# Changes made (compared to earlier version):
# - Use JSON structured context (less hallucination)
# - Query expansion (synonyms) for better recall
# - Deduplicate and rank search results, log scores
# - Collapse repeated sentences and repeated fallback phrases
# - Deterministic programmatic fallback when LLM fails but we have products
# - More timing/logging to help you profile slow parts
#
# Place next to your chroma_db folder and run as before:
#   uvicorn chatbotV3:app --reload

import uvicorn
import json
import re
import time
import asyncio
import os
import logging
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

app = FastAPI(debug=True, title="Superior Sounds Chatbot API", version="1.0.0")
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("chatbotV3")

# Embedding object used only for vectorstore interface
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load existing Chroma DB (created by embed_products.py)
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

# CORS for dev - tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domains in prod
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class ChatRequest(BaseModel):
    question: str

# LLM config - tuned for deterministic answers
llm = Ollama(
    model="qwen2.5:0.5b",  # keep yours or replace
    temperature=0.1,
    num_predict=500,
    num_ctx=2048,
    repeat_penalty=1.5,
    top_k=40,
    top_p=0.9
)

# Local synonym expansion (kept small & deterministic)
# If you want reuse, move to a small module both scripts import.
SYNONYM_EXPANSIONS = {
    'mic': 'microphone',
    'wireless mic': 'wireless microphone',
    'mics': 'microphones',
    'lights': 'lighting',
    'light': 'lighting',
    'speaker': 'speakers sound audio',
    'speakers': 'speakers sound audio',
    'dj': 'DJ turntable mixer',
    'wedding': 'event party reception',
    'outdoor': 'outside garden patio',
    'haze': 'haze fog smoke fogger haze machine',
    'fog': 'fog smoke haze machine'
}

def expand_with_synonyms(query: str) -> str:
    q = query.lower()
    for short, expanded in SYNONYM_EXPANSIONS.items():
        if short in q:
            q += " " + expanded
    return q

SYSTEM_PROMPT_TEMPLATE = """You are Superior Sounds' rental assistant. You MUST answer using ONLY the product information in the JSON array below.

IMPORTANT:
- The Product Information below is a JSON array of product objects with fields:
    title, price, description, url, category
- Use the exact title and price from the JSON if you reference a product.
- Include the product 'url' when you recommend any product.
- If you cannot find any relevant product, REPLY EXACTLY ONCE with:
    I could not find that item in our current inventory.
  (Exact punctuation, and do not add extra explanation.)
- Do NOT invent prices, specs, or products that are not present in the JSON.
- Keep replies concise (under ~200 words) and factual.

Product Information (JSON):
{context}
"""

def build_json_context(docs: List[Document], max_items: int = 8) -> str:
    """Return a JSON array (string) with product objects trimmed for length."""
    items = []
    seen_titles = set()
    for doc in docs:
        title = doc.metadata.get("title", "").strip()
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        price = doc.metadata.get("price", "").strip()
        url = doc.metadata.get("url", "").strip()
        category = doc.metadata.get("category", "").strip()
        # For description prefer metadata.description else doc.page_content
        desc = doc.metadata.get("description") or doc.page_content or ""
        if len(desc) > 300:
            desc = desc[:297] + "..."
        item = {"title": title, "price": price, "description": desc, "url": url, "category": category}
        items.append(item)
        if len(items) >= max_items:
            break
    return json.dumps(items, ensure_ascii=False, indent=2)

def collapse_duplicate_sentences(text: str) -> str:
    """Collapse immediately repeated sentences to a single occurrence."""
    # Split into sentences (keep punctuation)
    # This is a best-effort approach; it collapses consecutive identical sentences.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    filtered = []
    for s in sentences:
        if not filtered or s.strip() != filtered[-1].strip():
            filtered.append(s)
    return " ".join(filtered)

def programmatic_product_answer(docs: List[Document], max_items: int = 8) -> str:
    """Build a deterministic, factual answer from product metadata."""
    lines = []
    lines.append("Here are items from our inventory that match your request:")
    for doc in docs[:max_items]:
        t = doc.metadata.get("title", "Unknown")
        p = doc.metadata.get("price", "Call for pricing")
        desc = doc.metadata.get("description", "") or doc.page_content or ""
        if len(desc) > 150:
            desc = desc[:147] + "..."
        url = doc.metadata.get("url", "")
        lines.append(f"- {t} — ${p}/day. {desc} Book: {url}")
    lines.append("If you want a shorter list or a different category, tell me.")
    return "\n".join(lines)

async def enhanced_product_search(question: str, k: int = 10) -> List[Document]:
    """Search with query expansion; use scores when available and dedupe by URL."""
    expanded = expand_with_synonyms(question)
    logger.debug(f"Expanded search query: '{expanded}'")
    # Try similarity_search_with_score if available to inspect scores
    candidates = []
    try:
        scored = db.similarity_search_with_score(expanded, k=k * 3)
        # scored is a list of (Document, score) pairs
        logger.debug("Used similarity_search_with_score")
        for doc, score in scored:
            candidates.append((doc, score))
    except Exception:
        # fallback: similarity_search only
        docs = db.similarity_search(expanded, k=k * 3)
        candidates = [(d, None) for d in docs]
        logger.debug("Used similarity_search (no scores available)")

    # Deduplicate while preserving initial order and prefer higher-score if present
    unique = []
    seen_urls = set()
    # If scores present, sort by score if it seems higher-is-better (we will keep the order returned)
    for doc, score in candidates:
        url = doc.metadata.get("url", "").strip()
        title = doc.metadata.get("title", "").strip().lower()
        key = url or title
        if not key:
            key = title
        if key in seen_urls:
            continue
        seen_urls.add(key)
        unique.append(doc)
        if len(unique) >= k:
            break

    logger.debug(f"Search returned {len(unique)} unique results (requested {k})")
    # Log brief preview
    for i, d in enumerate(unique[:10], 1):
        logger.debug(f"  result {i}: {d.metadata.get('title','?')} | url={d.metadata.get('url','')}")
    return unique

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        return {"error": "Empty question"}

    try:
        search_start = time.time()
        products = await enhanced_product_search(question, k=8)
        search_time = time.time() - search_start
        logger.info(f"🔍 Search took {search_time:.2f}s — found {len(products)} products")

        # Build JSON context for LLM
        context_start = time.time()
        json_context = build_json_context(products, max_items=8)
        context_time = time.time() - context_start
        logger.debug(f"Context build took {context_time:.3f}s")
        logger.debug(f"Context preview: {json_context[:400]}")

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=json_context)
        full_prompt = f"{system_prompt}\n\nCustomer Question: {question}\n\nAssistant:"
        logger.debug(f"Full prompt length: {len(full_prompt)} chars")

        async def stream_response():
            # If there are no products, return a helpful fallback message immediately (deterministic)
            if not products:
                fallback = ("I could not find that item in our current inventory.")
                # This exact phrase respects your system rule; return it once and stop.
                yield fallback
                return

            # Query the LLM
            try:
                llm_start = time.time()
                logger.info("🧠 Calling LLM...")
                # Use async invocation; many LLM wrappers return a response object or a string
                response = await llm.ainvoke(
                    full_prompt,
                    temperature=0.1,
                    max_tokens=400,
                    stop=["Customer:", "Question:", "Assistant:"]
                )
                llm_time = time.time() - llm_start
                logger.info(f"🧠 LLM finished in {llm_time:.2f}s")

                # Extract text
                if hasattr(response, "content"):
                    text = response.content
                else:
                    text = str(response)

                if not text or text.strip() == "":
                    # empty response — fallback to programmatic answer
                    logger.warning("LLM returned empty text; using deterministic fallback")
                    text = programmatic_product_answer(products)

                # collapse repeated sentences (very common source of the repetition problem)
                text = collapse_duplicate_sentences(text)

                # If LLM says it cannot find something (but we DO have products), override with deterministic answer
                if "could not find that item in our current inventory" in text.lower():
                    logger.warning("LLM refused to use provided inventory; using programmatic deterministic answer.")
                    text = programmatic_product_answer(products)

                # Final safety: strip out mentions we explicitly disallow
                banned_phrases = ["ebay", "amazon", "buy online", "purchase online"]
                lowered = text.lower()
                if any(bp in lowered for bp in banned_phrases):
                    logger.warning("LLM included banned phrase; replacing with deterministic answer.")
                    text = programmatic_product_answer(products)

                # Stream the final (cleaned) text word-by-word
                words = text.split()
                for i, w in enumerate(words):
                    yield w + (" " if i < len(words) - 1 else "")
                    # small sleep to allow client to show typing — adjustable
                    if i % 6 == 0:
                        await asyncio.sleep(0.01)

            except Exception as e:
                logger.exception("LLM invocation error")
                # Fallback to deterministic product answer if we have products
                if products:
                    fallback = programmatic_product_answer(products[:6])
                else:
                    fallback = ("I'm having technical difficulties. Please try again later or contact Superior Sounds at https://twice.shop/superiorsounds1/shop")
                # stream fallback
                for word in fallback.split():
                    yield word + " "
                    await asyncio.sleep(0.01)

        # Return a streaming plain text response (your JS widget expects text stream)
        return StreamingResponse(stream_response(), media_type="text/plain")

    except Exception as e:
        logger.exception("General error in /chat")
        return {"error": "Something went wrong on the server. Please try again."}

# Health and widget endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "qwen2.5:0.5b",
        "cors_enabled": True
    }

@app.get("/widget-config")
async def widget_config():
    return {
        "api_endpoint": "http://localhost:8000",
        "brand_name": "Superior Sounds",
        "brand_color": "#7c3aed",
        "welcome_message": "👋 Hi! I'm here to help you find the perfect sound and lighting equipment for your event. What are you planning?"
    }

if __name__ == "__main__":
    uvicorn.run("chatbotV3:app", host="127.0.0.1", port=8000, reload=True)
