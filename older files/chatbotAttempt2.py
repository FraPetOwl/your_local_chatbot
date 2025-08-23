# chatbotAttempt2.py
# This script sets up a FastAPI server to handle chat requests using a rule-based product matcher 
# - good as backup for when LLMs fail.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import asyncio
import json
from langchain_community.vectorstores import Chroma
from embed_products import db
from fastapi.responses import StreamingResponse
import logging

app = FastAPI(debug=True)
logging.basicConfig(level=logging.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class FastProductMatcher:
    """Ultra-fast rule-based product matching for common queries"""
    
    def __init__(self):
        self.keywords = {
            'microphone': ['mic', 'microphone', 'vocal', 'singing'],
            'speaker': ['speaker', 'sound', 'audio', 'pa system'],
            'lighting': ['light', 'led', 'stage light', 'color', 'lighting'],
            'dj': ['dj', 'turntable', 'mixer', 'controller', 'cdj'],
            'projector': ['projector', 'screen', 'display', 'presentation'],
            'stage': ['stage', 'platform', 'riser', 'steps']
        }
    
    def quick_match(self, question: str, products: list) -> str:
        """Generate fast responses for common queries"""
        question_lower = question.lower()
        
        # Count category matches
        category_scores = {}
        for category, keywords in self.keywords.items():
            score = sum(1 for kw in keywords if kw in question_lower)
            if score > 0:
                category_scores[category] = score
        
        if not category_scores:
            return None
        
        # Find products matching top category
        top_category = max(category_scores, key=category_scores.get)
        matching_products = []
        
        for product in products:
            content = product.page_content.lower()
            title = product.metadata.get('title', '').lower()
            
            if any(kw in content or kw in title for kw in self.keywords[top_category]):
                matching_products.append(product)
        
        # Generate response
        if matching_products:
            response = f"Here are our {top_category} rentals:\n\n"
            for i, product in enumerate(matching_products[:4]):  # Limit to 4
                title = product.metadata.get('title', 'Unknown')
                price = product.metadata.get('price', 'N/A')
                url = product.metadata.get('url', '')
                description = product.page_content[:150]
                
                response += f"{i+1}. **{title}** - ${price}/day\n"
                response += f"   {description}...\n"
                response += f"   Link: {url}\n\n"
            
            return response
        
        return None

# Initialize fast matcher
fast_matcher = FastProductMatcher()

# Simple template-based responses
def generate_template_response(question: str, products: list) -> str:
    """Generate responses using simple templates"""
    
    if len(products) == 0:
        return "I couldn't find any products matching your request. Could you be more specific about what type of equipment you need?"
    
    # Quick categorization
    categories = {}
    for product in products[:6]:  # Limit processing
        title = product.metadata.get('title', '').lower()
        if any(word in title for word in ['mic', 'microphone']):
            categories.setdefault('microphones', []).append(product)
        elif any(word in title for word in ['speaker', 'pa']):
            categories.setdefault('speakers', []).append(product)
        elif any(word in title for word in ['light', 'led']):
            categories.setdefault('lighting', []).append(product)
        elif any(word in title for word in ['dj', 'mixer', 'turntable']):
            categories.setdefault('dj_equipment', []).append(product)
        else:
            categories.setdefault('other', []).append(product)
    
    # Generate response based on categories
    response = "I found these rental options for you:\n\n"
    
    for category, items in categories.items():
        if category != 'other':
            cat_name = category.replace('_', ' ').title()
            response += f"**{cat_name}:**\n"
            
            for item in items[:3]:  # Max 3 per category
                title = item.metadata.get('title', 'Unknown')
                price = item.metadata.get('price', 'N/A')
                url = item.metadata.get('url', '')
                
                response += f"• {title} - ${price}/day\n"
                response += f"  {url}\n"
            response += "\n"
    
    response += "Need more details about any of these items? Just ask!"
    return response

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question
    
    try:
        start_time = time.time()
        
        # Fast similarity search
        raw_docs = db.similarity_search(question, k=10)
        search_time = time.time() - start_time
        print(f"🔍 Search took {search_time:.2f}s")
        
        # Quick deduplication
        unique_products = []
        seen_titles = set()
        
        for doc in raw_docs:
            title = doc.metadata.get("title", "")
            if title not in seen_titles:
                unique_products.append(doc)
                seen_titles.add(title)
        
        # Try fast matching first
        response_start = time.time()
        quick_response = fast_matcher.quick_match(question, unique_products)
        
        if quick_response:
            final_response = quick_response
        else:
            # Fall back to template response
            final_response = generate_template_response(question, unique_products)
        
        response_time = time.time() - response_start
        total_time = time.time() - start_time
        
        print(f"⚡ Response generation: {response_time:.2f}s")
        print(f"🏁 Total time: {total_time:.2f}s")
        
        # Stream the response for better UX
        async def fast_stream():
            words = final_response.split()
            for i, word in enumerate(words):
                yield word + " "
                if i % 5 == 0:  # Every 5 words
                    await asyncio.sleep(0.01)
        
        return StreamingResponse(fast_stream(), media_type="text/plain")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": "Something went wrong. Please try again."}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "type": "rule_based"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)