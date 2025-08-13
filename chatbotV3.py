# chatbotV4_optimized.py
# Optimized version with better responses and faster performance

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbotV4")

app = FastAPI(debug=True, title="Superior Sounds Chatbot API", version="4.0.0")

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

# Initialize components
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

try:
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )
    logger.info("✅ Database loaded successfully")
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

# Optimized LLM config for speed
llm = Ollama(
    model="qwen2.5:0.5b",
    temperature=0.1,
    num_predict=300,  # Reduced for speed
    num_ctx=1024,     # Reduced context for speed
    repeat_penalty=1.2,
    timeout=15        # 15 second timeout
)

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

def build_enhanced_context(docs: List[Document], max_items: int = 6) -> List[dict]:
    """Build enhanced context with better product data."""
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
        description = doc.metadata.get("description") or doc.page_content or ""
        
        # Keep more description for better responses
        if len(description) > 300:
            description = description[:297] + "..."
            
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
    
    base_rules = """You are Superior Sounds' rental assistant. RULES:
1. Use ONLY the products listed below
2. We rent equipment by the day - mention this clearly
3. Always include "Book:" followed by the URL when recommending products
4. Use exact product titles and prices from the data
5. Be helpful and mention relevant product categories"""

    intent_instructions = {
        'event_package': 'Suggest multiple items that work well together for the event type.',
        'recommendation': 'Provide detailed recommendations with reasoning.',
        'budget_focused': 'Highlight the most affordable options and mention prices clearly.',
        'product_search': 'List relevant products with key details and booking links.',
        'general': 'Be helpful and informative about our rental inventory.'
    }
    
    instruction = intent_instructions.get(intent, intent_instructions['general'])
    
    context_json = json.dumps(context, indent=2)
    
    return f"""{base_rules}

{instruction}

EXAMPLE RESPONSES:
- "Yes! We have wireless microphones like the Sennheiser XSW 1-825 ($75/day). Book: [URL]"
- "For karaoke nights, you'll need microphones and a mixer. We have the Sennheiser wireless mic ($75/day) and Yamaha mixer ($150/day). Book: [URL1] [URL2]"

PRODUCTS:
{context_json}"""

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
    
    # Ensure URLs are properly formatted
    if products and '[url]' in response_lower:
        # Replace [URL] placeholders with actual URLs
        for product in products:
            if product.get('url'):
                response = response.replace('[URL]', product['url'], 1)
    
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
        # Detect intent for better responses
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
                else:
                    # Create intent-based prompt
                    system_prompt = create_intent_based_prompt(products, intent)
                    full_prompt = f"{system_prompt}\n\nCustomer: {question}\n\nAssistant:"
                    
                    logger.info("🧠 Querying LLM...")
                    llm_start = time.time()
                    
                    try:
                        # Use invoke with timeout
                        llm_response = llm.invoke(full_prompt)
                        llm_time = time.time() - llm_start
                        logger.info(f"🧠 LLM: {llm_time:.2f}s")
                        
                        # Extract response
                        if hasattr(llm_response, 'content'):
                            response_text = llm_response.content
                        else:
                            response_text = str(llm_response)
                        
                        # Validate and enhance
                        validated = validate_and_enhance_response(response_text, products, question)
                        
                        if validated:
                            response = validated
                            logger.info("✅ Using LLM response")
                        else:
                            response = create_smart_fallback(products, question, intent)
                            logger.info("⚠️ Using smart fallback")
                            
                    except Exception as e:
                        logger.error(f"LLM error: {e}")
                        response = create_smart_fallback(products, question, intent)
                
                # Stream response with faster timing
                words = response.split()
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")
                    # Faster streaming
                    if i % 8 == 0:
                        await asyncio.sleep(0.01)
                        
            except Exception as e:
                logger.exception("Streaming error")
                error_msg = "Sorry, I'm having technical difficulties. Please try again."
                for word in error_msg.split():
                    yield word + " "
                    await asyncio.sleep(0.01)
        
        return StreamingResponse(stream_response(), media_type="text/plain")
        
    except Exception as e:
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
        "model": "qwen2.5:0.5b"
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
    uvicorn.run("chatbotV4_optimized:app", host="127.0.0.1", port=8000, reload=True)