import requests
import time
import csv
import tiktoken  # for token counting (pip install tiktoken)

API_URL = "http://127.0.0.1:8000/chat"

# Example test questions
test_questions = [
    "Do you have any wireless microphones for rent?",
    "What lights would work best for an outdoor wedding?",
    "Show me your most affordable DJ speaker.",
    "Do you have smoke or haze machines?",
    "Which moving head lights do you rent?",
    "I need audio and lighting for a 200-person party. What do you recommend?",
    "Suggest a full setup for a small corporate event.",
    "What would I need for a karaoke night?",
    "Do you sell these products?",
    "How much is delivery to my house?",
    "What’s the warranty on these lights?",
    "Give me your top 3 lighting options for a nightclub event.",
    "List your DJ controllers with prices and booking links.",
    "Show me all your party packages and include prices.",
    "Tell me about the Superior Sounds T-shirt you have for sale.",
    "What can I rent for my retirement party in a large space?"
]

# OpenAI-style encoder for token count
enc = tiktoken.get_encoding("cl100k_base") # same as gpt-3.5-turbo

def count_tokens(text):
    return len(enc.encode(text))

results = []

for question in test_questions:
    print(f"Testing: {question}")
    
    # Record full latency
    start_time = time.time()
    
    # Streaming request
    with requests.post(API_URL, json={"question": question, "style": "concise"}, stream=True) as resp:
        resp.raise_for_status()
        
        response_chunks = []
        first_token_time = None
        
        for chunk in resp.iter_lines():
            if chunk:
                decoded = chunk.decode("utf-8")
                if not first_token_time:
                    first_token_time = time.time() - start_time
                response_chunks.append(decoded)
        
        full_response = "".join(response_chunks)
    
    latency = time.time() - start_time
    token_count = count_tokens(full_response)
    
    results.append({
        "question": question,
        "response": full_response,
        "latency_sec": round(latency, 2),
        "first_token_sec": round(first_token_time, 2) if first_token_time else None,
        "token_count": token_count,
        "accuracy_grade": ""  # fill manually later
    })

# Save results to CSV
with open("qa_results_detailed.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("✅ QA run complete. Results saved to qa_results_detailed.csv")
