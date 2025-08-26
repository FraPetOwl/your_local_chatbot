# simple_llm_test.py - Standalone LLM test
import os
import time

try:
    from llama_cpp import Llama
    print("✅ llama_cpp imported successfully")
except ImportError as e:
    print(f"❌ Failed to import llama_cpp: {e}")
    exit(1)

# Use same settings as your main app
llm_model_filename = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
llm_repo_id = "bartowski/Llama-3.2-3B-Instruct-GGUF"

print("Loading LLM...")
start_time = time.time()

try:
    llm = Llama.from_pretrained(
        repo_id=llm_repo_id,
        filename=llm_model_filename,
        n_ctx=2048,
        n_threads=6,
        n_batch=32,
        verbose=False
    )
    load_time = time.time() - start_time
    print(f"✅ LLM loaded in {load_time:.2f}s")
except Exception as e:
    print(f"❌ LLM loading failed: {e}")
    exit(1)

# Test simple call
print("\nTesting LLM call...")
test_start = time.time()

try:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Say exactly: 'LLM is working'"}],
        temperature=0.1,
        max_tokens=20,
        stream=False
    )
    
    test_time = time.time() - test_start
    
    if "choices" in response and len(response["choices"]) > 0:
        content = response["choices"][0]["message"]["content"]
        print(f"✅ LLM responded in {test_time:.2f}s: '{content}'")
    else:
        print(f"❌ Unexpected response format: {response}")
        
except Exception as e:
    test_time = time.time() - test_start
    print(f"❌ LLM call failed after {test_time:.2f}s: {e}")

print("\nTest complete.")