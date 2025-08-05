from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

app = FastAPI()

# CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

# Initialize vector store
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)

# Load LLM
llm = Ollama(model="phi3:mini", temperature=0.0, num_predict=500)  # Set temperature to 0 for more deterministic responses

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question

    # Retrieve context from Chroma
    start = time.time()
    # Similarity search
    raw_docs = db.similarity_search(question, k=15)

    # Deduplicate by metadata['title'] + optionally URL or price
    seen_titles = set()
    deduped_docs = []
    for doc in raw_docs:
        title = doc.metadata.get("title", "").strip().lower()
        if title and title not in seen_titles:
            seen_titles.add(title)
            deduped_docs.append(doc)
            
    max_context_length = 3000  # 
    context = ""
    for doc in deduped_docs:
        next_entry = f"\n---\n{doc.page_content}"
        if len(context) + len(next_entry) > max_context_length:
            break
        context += next_entry


    print(f"🔍 Chroma search took {round(time.time() - start, 2)}s")
    print(f"🔍 Retrieved {len(deduped_docs)} documents for context.")
    print(f"Context:\n{context}")

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
        print(f"\n🧠 Starting generation...")
        llm_start = time.time()
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"{question}\n\nContext:\n{context}")
        ]
        raw = llm.invoke(messages)
        print(f"🧠 LLM generation took {round(time.time() - llm_start, 2)}s")
        return {
            "question": question,
            "context": context,
            "answer": raw.strip()
        }
    except Exception as e:
        return {"error": str(e)}

# Run locally if needed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
