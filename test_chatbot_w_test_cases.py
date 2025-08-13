# test_chatbot.py
# Test script to validate chatbot improvements
# Run this to test the key issues found in the chat log

import asyncio
import aiohttp
import json
import time

async def test_endpoint(session, question, expected_keywords=None):
    """Test a single question against the chatbot API."""
    print(f"\n🔍 Testing: '{question}'")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        async with session.post(
            'http://localhost:8000/chat',
            json={"question": question, "style": "concise"},
            timeout=30
        ) as response:
            if response.status != 200:
                print(f"❌ HTTP {response.status}: {await response.text()}")
                return False
            
            # Stream the response
            full_response = ""
            async for chunk in response.content.iter_any():
                chunk_text = chunk.decode('utf-8')
                full_response += chunk_text
                print(chunk_text, end='', flush=True)
    
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False
    
    duration = time.time() - start_time
    print(f"\n\n⏱️ Response time: {duration:.2f}s")
    
    # Validate response
    issues = []
    response_lower = full_response.lower()
    
    # Check for common problems
    if "couldn't find that item" in response_lower and len(full_response) < 100:
        issues.append("❌ Generic 'not found' response when products likely exist")
    
    if any(bad_phrase in response_lower for bad_phrase in ['yuan', 'rmb', 'ebay', 'amazon']):
        issues.append("❌ Contains banned phrases")
    
    if any(word in response_lower for word in ['buy', 'purchase', 'sell']) and 'rent' not in response_lower:
        issues.append("❌ Mentions purchasing instead of rental")
    
    # More flexible keyword checking - only flag if missing critical keywords
    if expected_keywords:
        critical_missing = []
        for keyword in expected_keywords:
            if keyword.lower() in ['/day', 'book:']:  # Always required
                if keyword.lower() not in response_lower:
                    critical_missing.append(keyword)
            elif keyword.lower() in ['microphone', 'speaker', 'light', 'lighting']:  # Core product terms
                # Check for the term or common variations
                variations = {
                    'microphone': ['mic', 'microphone'],
                    'speaker': ['speaker', 'sound', 'audio'],
                    'light': ['light', 'lighting', 'uplight', 'led'],
                    'lighting': ['light', 'lighting', 'uplight', 'led']
                }
                if keyword.lower() in variations:
                    if not any(var in response_lower for var in variations[keyword.lower()]):
                        critical_missing.append(keyword)
                elif keyword.lower() not in response_lower:
                    critical_missing.append(keyword)
        
        if critical_missing:
            issues.append(f"❌ Missing critical keywords: {critical_missing}")
    
    # Print validation results
    if issues:
        print("🚨 ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ Response looks good!")
        return True

async def run_tests():
    """Run all test cases from the problematic chat log."""
    
    test_cases = [
        {
            "question": "Do you have any wireless microphones for rent?",
            "expected_keywords": ["microphone", "wireless", "/day", "Book:"],
            "description": "Basic microphone search - should find actual mics"
        },
        {
            "question": "What lights would work best for an outdoor wedding?", 
            "expected_keywords": ["light", "LED", "uplight", "/day"],
            "description": "Wedding lighting - should suggest appropriate lights"
        },
        {
            "question": "Show me your most affordable DJ speaker",
            "expected_keywords": ["speaker", "$", "/day", "DJ"],
            "description": "Budget DJ speaker search"
        },
        {
            "question": "Do you have smoke or haze machines?",
            "expected_keywords": ["smoke", "haze", "machine", "/day"],
            "description": "Effects equipment search"
        },
        {
            "question": "Which moving head lights do you rent?",
            "expected_keywords": ["moving", "head", "light", "/day"],
            "description": "Specific lighting equipment"
        },
        {
            "question": "I need audio and lighting for a 200-person party. What do you recommend?",
            "expected_keywords": ["audio", "lighting", "speaker", "light"],
            "description": "Complete system recommendation"
        },
        {
            "question": "What would I need for a karaoke night?",
            "expected_keywords": ["microphone", "mixer", "speaker"],
            "description": "Event-specific equipment needs"
        }
    ]
    
    print("🚀 CHATBOT IMPROVEMENT VALIDATION")
    print("=" * 60)
    print("Testing the key problem areas from the chat log...\n")
    
    async with aiohttp.ClientSession() as session:
        passed = 0
        total = len(test_cases)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 TEST {i}/{total}: {test_case['description']}")
            
            success = await test_endpoint(
                session, 
                test_case["question"], 
                test_case.get("expected_keywords")
            )
            
            if success:
                passed += 1
            
            # Brief pause between tests
            await asyncio.sleep(1)
    
        print(f"\n🎯 FINAL RESULTS: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All tests passed! Improvements are working.")
        else:
            print(f"⚠️ {total - passed} tests failed. Review the issues above.")

async def test_health_check():
    """Test that the API is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health') as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ API is running - Version: {data.get('version', 'unknown')}")
                    return True
                else:
                    print(f"❌ Health check failed: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure the chatbot server is running with: uvicorn chatbotV4:app --reload")
        return False

if __name__ == "__main__":
    print("🔧 Superior Sounds Chatbot Test Suite")
    print("=" * 50)
    
    # Check if API is accessible
    if not asyncio.run(test_health_check()):
        exit(1)
    
    # Run the test suite
    asyncio.run(run_tests())