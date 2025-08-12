# chatbotV4.py
# Enhanced Superior Sounds — FastAPI chatbot API with improved LLM responses
# Key improvements:
# - Better prompt engineering for more natural responses
# - Improved context filtering and ranking
# - Enhanced query understanding with intent detection
# - Better error handling and response validation
# - Configurable response styles (concise vs detailed)
# - Response quality metrics and logging

import uvicorn
import json
import re
import time
import asyncio
import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chatbotV4")

app = FastAPI(debug=True, title="Superior Sounds Chatbot API", version="4.0.0")

class ResponseStyle(str, Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"

class QueryIntent(str, Enum):
    PRODUCT_SEARCH = "product_search"
    PRICE_INQUIRY = "price_inquiry" 
    AVAILABILITY = "availability"
    COMPARISON = "comparison"
    GENERAL_INFO = "general_info"

@dataclass
class SearchResult:
    document: Document
    score: Optional[float]
    relevance_boost: float = 1.0

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    style: ResponseStyle = ResponseStyle.CONCISE
    max_products: int = Field(default=6, ge=1, le=15)

class ChatConfig:
    # Enhanced synonym mappings with context
    SYNONYM_EXPANSIONS = {
        # Audio equipment
        'mic': 'microphone vocal handheld wireless',
        'mics': 'microphones vocal handheld wireless',
        'speaker': 'speakers sound audio PA system monitor',
        'speakers': 'speakers sound audio PA system monitors',
        'pa': 'PA system public address speakers sound',
        'sound system': 'PA speakers audio equipment',
        
        # Lighting
        'lights': 'lighting LED uplight spotlight wash',
        'light': 'lighting LED uplight spotlight wash',
        'uplight': 'LED uplight color wash ambient',
        'spotlight': 'spot light beam focused lighting',
        
        # DJ equipment
        'dj': 'DJ turntable mixer controller deck',
        'turntable': 'DJ turntable vinyl deck controller',
        'mixer': 'audio mixer DJ mixing console board',
        
        # Effects
        'haze': 'haze fog smoke atmosphere effect machine',
        'fog': 'fog haze smoke atmosphere effect machine',
        'smoke': 'smoke fog haze atmosphere effect',
        
        # Event types
        'wedding': 'wedding reception ceremony event party',
        'party': 'party celebration event reception gathering',
        'outdoor': 'outdoor outside garden patio event',
        'corporate': 'corporate business professional meeting',
        
        # Technical terms
        'wireless': 'wireless cordless battery powered',
        'portable': 'portable mobile battery powered compact',
        'professional': 'professional pro grade commercial quality'
    }
    
    # Price-related keywords for intent detection
    PRICE_KEYWORDS = ['cost', 'price', 'pricing', 'expensive', 'cheap', 'budget', 'rate', 'fee']
    AVAILABILITY_KEYWORDS = ['available', 'in stock', 'rent', 'rental', 'book', 'reserve']
    COMPARISON_KEYWORDS = ['vs', 'versus', 'compare', 'difference', 'better', 'best', 'recommend']

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

llm = Ollama(
    model="qwen2.5:0.5b",
    temperature=0.2,  # Slightly higher for more natural responses
    num_predict=600,  # Increased for better responses
    num_ctx=3072,     # Increased context window
    repeat_penalty=1.3,
    top_k=50,
    top_p=0.95
)

class QueryProcessor:
    @staticmethod
    def detect_intent(query: str) -> QueryIntent:
        """Detect the primary intent of the user's query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ChatConfig.PRICE_KEYWORDS):
            return QueryIntent.PRICE_INQUIRY
        elif any(word in query_lower for word in ChatConfig.AVAILABILITY_KEYWORDS):
            return QueryIntent.AVAILABILITY
        elif any(word in query_lower for word in ChatConfig.COMPARISON_KEYWORDS):
            return QueryIntent.COMPARISON
        elif any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
            return QueryIntent.GENERAL_INFO
        else:
            return QueryIntent.PRODUCT_SEARCH

    @staticmethod
    def expand_with_synonyms(query: str) -> str:
        """Enhanced query expansion with better context preservation."""
        expanded_terms = []
        query_lower = query.lower()
        
        # Add original query
        expanded_terms.append(query)
        
        # Add synonym expansions
        for term, expansions in ChatConfig.SYNONYM_EXPANSIONS.items():
            if term in query_lower:
                expanded_terms.append(expansions)
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in expanded_terms:
            words = term.split()
            for word in words:
                if word not in seen:
                    unique_terms.append(word)
                    seen.add(word)
        
        return ' '.join(unique_terms)

    @staticmethod
    def extract_product_categories(query: str) -> List[str]:
        """Extract likely product categories from query."""
        categories = []
        query_lower = query.lower()
        
        category_keywords = {
            'audio': ['microphone', 'mic', 'speaker', 'sound', 'pa', 'audio'],
            'lighting': ['light', 'lighting', 'uplight', 'spotlight', 'led'],
            'dj': ['dj', 'turntable', 'mixer', 'controller'],
            'effects': ['haze', 'fog', 'smoke', 'effect'],
            'av': ['projector', 'screen', 'video', 'display']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                categories.append(category)
        
        return categories or ['equipment']  # Default fallback

class ContextBuilder:
    @staticmethod
    def rank_results(results: List[Tuple[Document, Optional[float]]], 
                    query: str, intent: QueryIntent) -> List[SearchResult]:
        """Enhanced result ranking with intent-based boosting."""
        search_results = []
        query_lower = query.lower()
        
        for doc, score in results:
            relevance_boost = 1.0
            title = doc.metadata.get('title', '').lower()
            category = doc.metadata.get('category', '').lower()
            description = doc.metadata.get('description', '').lower()
            
            # Boost based on query intent
            if intent == QueryIntent.PRICE_INQUIRY:
                # Boost if price is prominently mentioned
                price = doc.metadata.get('price', '')
                if price and price.lower() not in ['call', 'tbd', 'n/a']:
                    relevance_boost += 0.2
            
            elif intent == QueryIntent.COMPARISON:
                # Boost popular/professional items for comparisons
                if any(word in title for word in ['professional', 'pro', 'deluxe']):
                    relevance_boost += 0.3
            
            # Boost exact matches
            query_words = set(query_lower.split())
            title_words = set(title.split())
            if query_words.intersection(title_words):
                match_ratio = len(query_words.intersection(title_words)) / len(query_words)
                relevance_boost += match_ratio * 0.5
            
            # Category relevance boost
            categories = QueryProcessor.extract_product_categories(query)
            if any(cat in category for cat in categories):
                relevance_boost += 0.2
            
            search_results.append(SearchResult(doc, score, relevance_boost))
        
        # Sort by combined score
        search_results.sort(key=lambda x: (x.score or 0) * x.relevance_boost, reverse=True)
        return search_results

    @staticmethod
    def build_enhanced_context(results: List[SearchResult], 
                              intent: QueryIntent, 
                              max_items: int = 8) -> Dict:
        """Build structured context with metadata for better LLM understanding."""
        products = []
        seen_titles = set()
        
        for result in results[:max_items]:
            doc = result.document
            title = doc.metadata.get("title", "").strip()
            
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            
            price = doc.metadata.get("price", "").strip()
            url = doc.metadata.get("url", "").strip()
            category = doc.metadata.get("category", "").strip()
            description = doc.metadata.get("description", "") or doc.page_content or ""
            
            # Trim description
            if len(description) > 200:
                description = description[:197] + "..."
            
            product = {
                "title": title,
                "price": price,
                "category": category,
                "description": description,
                "url": url,
                "relevance_score": round(result.relevance_boost, 2)
            }
            products.append(product)
        
        return {
            "query_intent": intent.value,
            "total_results": len(products),
            "products": products
        }

class PromptManager:
    @staticmethod
    def get_system_prompt(intent: QueryIntent, style: ResponseStyle) -> str:
        """Generate intent and style-specific system prompts."""
        
        base_rules = """You are Superior Sounds' helpful rental assistant. Follow these rules:

1. Use ONLY the product information provided in the JSON context below
2. Always include the product URL when recommending specific items
3. Keep prices and product names exactly as shown in the JSON
4. If no relevant products exist, respond: "I couldn't find that item in our current inventory."
5. Stay focused on Superior Sounds' rental inventory"""

        style_instructions = {
            ResponseStyle.CONCISE: "Keep responses brief and to-the-point (1-2 sentences per product).",
            ResponseStyle.DETAILED: "Provide comprehensive information including features and use cases.",
            ResponseStyle.TECHNICAL: "Include technical specifications and professional recommendations."
        }

        intent_instructions = {
            QueryIntent.PRODUCT_SEARCH: "Help find the best products matching their needs.",
            QueryIntent.PRICE_INQUIRY: "Focus on pricing information and value propositions.",
            QueryIntent.AVAILABILITY: "Emphasize availability and booking information.",
            QueryIntent.COMPARISON: "Compare products highlighting key differences and recommendations.",
            QueryIntent.GENERAL_INFO: "Provide helpful information about our services and equipment."
        }

        return f"""{base_rules}

Response Style: {style_instructions.get(style, style_instructions[ResponseStyle.CONCISE])}
Query Intent: {intent_instructions.get(intent, intent_instructions[QueryIntent.PRODUCT_SEARCH])}

Product Information (JSON):
{{context}}"""

class ResponseValidator:
    @staticmethod
    def validate_response(response: str, products: List[Dict]) -> str:
        """Validate and clean LLM response."""
        if not response or len(response.strip()) < 10:
            return None
        
        # Remove banned phrases
        banned_phrases = [
            "ebay", "amazon", "buy online", "purchase online", 
            "alibaba", "walmart", "target"
        ]
        
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in banned_phrases):
            logger.warning("Response contained banned phrases")
            return None
        
        # Check if response uses provided products
        if products:
            product_mentioned = False
            for product in products:
                title_words = product.get('title', '').lower().split()
                if any(word in response_lower for word in title_words if len(word) > 3):
                    product_mentioned = True
                    break
            
            if not product_mentioned:
                logger.warning("Response didn't reference provided products")
                return None
        
        return response.strip()

    @staticmethod
    def generate_fallback_response(products: List[Dict], intent: QueryIntent) -> str:
        """Generate deterministic fallback response."""
        if not products:
            return "I couldn't find that item in our current inventory. Please try a different search or contact us for more options."
        
        if intent == QueryIntent.PRICE_INQUIRY:
            lines = ["Here's pricing for items that match your request:"]
            for product in products[:5]:
                title = product.get('title', 'Unknown')
                price = product.get('price', 'Call for pricing')
                url = product.get('url', '')
                lines.append(f"• {title}: ${price}/day - Book at: {url}")
        
        elif intent == QueryIntent.COMPARISON:
            lines = ["Here are some options to compare:"]
            for i, product in enumerate(products[:4], 1):
                title = product.get('title', 'Unknown')
                price = product.get('price', 'N/A')
                desc = product.get('description', '')[:100] + "..." if len(product.get('description', '')) > 100 else product.get('description', '')
                lines.append(f"{i}. {title} (${price}/day) - {desc}")
        
        else:  # Default product listing
            lines = ["Here are items from our inventory that match your request:"]
            for product in products[:6]:
                title = product.get('title', 'Unknown')
                price = product.get('price', 'Call for pricing')
                url = product.get('url', '')
                lines.append(f"• {title} - ${price}/day. Book: {url}")
        
        lines.append("\nNeed help choosing? Feel free to ask more specific questions!")
        return "\n".join(lines)

async def enhanced_product_search(question: str, k: int = 12) -> List[Tuple[Document, Optional[float]]]:
    """Enhanced search with better query processing."""
    try:
        # Expand query
        expanded = QueryProcessor.expand_with_synonyms(question)
        logger.debug(f"Expanded query: '{expanded}'")
        
        # Try scored search first
        try:
            results = db.similarity_search_with_score(expanded, k=k)
            logger.debug(f"Used similarity_search_with_score, found {len(results)} results")
            return results
        except Exception:
            # Fallback to regular search
            docs = db.similarity_search(expanded, k=k)
            logger.debug(f"Used similarity_search fallback, found {len(docs)} results")
            return [(doc, None) for doc in docs]
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    start_time = time.time()
    
    try:
        # Process query
        intent = QueryProcessor.detect_intent(question)
        logger.info(f"🎯 Detected intent: {intent.value}")
        
        # Search products
        search_start = time.time()
        raw_results = await enhanced_product_search(question, k=15)
        search_time = time.time() - search_start
        
        # Rank and filter results
        ranked_results = ContextBuilder.rank_results(raw_results, question, intent)
        context_data = ContextBuilder.build_enhanced_context(
            ranked_results, intent, req.max_products
        )
        
        logger.info(f"🔍 Search: {search_time:.2f}s, found {len(context_data['products'])} products")
        
        async def stream_response():
            products = context_data['products']
            
            if not products:
                fallback = ResponseValidator.generate_fallback_response([], intent)
                for word in fallback.split():
                    yield word + " "
                    await asyncio.sleep(0.02)
                return
            
            try:
                # Build prompt
                system_prompt = PromptManager.get_system_prompt(intent, req.style)
                context_json = json.dumps(context_data, ensure_ascii=False, indent=2)
                full_prompt = f"{system_prompt.format(context=context_json)}\n\nCustomer Question: {question}\n\nAssistant:"
                
                # Query LLM
                llm_start = time.time()
                logger.info("🧠 Querying LLM...")
                
                response = await llm.ainvoke(
                    full_prompt,
                    temperature=0.2,
                    max_tokens=500,
                    stop=["Customer:", "Question:", "Assistant:"]
                )
                
                llm_time = time.time() - llm_start
                logger.info(f"🧠 LLM responded in {llm_time:.2f}s")
                
                # Extract and validate response
                text = response.content if hasattr(response, 'content') else str(response)
                validated_text = ResponseValidator.validate_response(text, products)
                
                if not validated_text:
                    logger.warning("LLM response failed validation, using fallback")
                    validated_text = ResponseValidator.generate_fallback_response(products, intent)
                
                # Stream response
                words = validated_text.split()
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) - 1 else "")
                    if i % 8 == 0:  # Slightly faster streaming
                        await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.exception("LLM invocation failed")
                fallback = ResponseValidator.generate_fallback_response(products, intent)
                for word in fallback.split():
                    yield word + " "
                    await asyncio.sleep(0.01)
        
        return StreamingResponse(stream_response(), media_type="text/plain")
        
    except Exception as e:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        total_time = time.time() - start_time
        logger.info(f"⏱️ Total request time: {total_time:.2f}s")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "4.0.0",
        "model": "qwen2.5:0.5b",
        "features": ["intent_detection", "response_validation", "enhanced_search"]
    }

@app.get("/widget-config")
async def widget_config():
    return {
        "api_endpoint": "http://localhost:8000",
        "brand_name": "Superior Sounds",
        "brand_color": "#7c3aed",
        "welcome_message": "👋 Hi! I'm here to help you find the perfect sound and lighting equipment for your event. What are you planning?",
        "response_styles": [
            {"value": "concise", "label": "Quick Answers"},
            {"value": "detailed", "label": "Detailed Info"},
            {"value": "technical", "label": "Technical Specs"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run("chatbotV4:app", host="127.0.0.1", port=8000, reload=True)