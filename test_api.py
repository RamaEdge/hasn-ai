#!/usr/bin/env python3
"""
Test script for Brain-Inspired Neural Network API
Demonstrates all major endpoints and functionality
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
API_BASE = "http://localhost:8000"

def test_api_endpoint(endpoint, method="GET", data=None, description=""):
    """Test a single API endpoint"""
    url = f"{API_BASE}{endpoint}"
    print(f"\n🧪 Testing: {description}")
    print(f"📡 {method} {endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('message', 'OK')}")
            return result
        else:
            print(f"❌ Error: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"🔌 Connection failed - make sure API server is running at {API_BASE}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Run comprehensive API tests"""
    print("🧠 Brain-Inspired Neural Network API Test Suite")
    print("=" * 50)
    
    # Test 1: Root endpoint
    root_result = test_api_endpoint(
        "/", 
        description="Root endpoint - API information"
    )
    
    # Test 2: Health check
    health_result = test_api_endpoint(
        "/health", 
        description="Health check - system status"
    )
    
    # Test 3: Neural pattern processing
    pattern_data = {
        "pattern": {
            "0": {"0": True, "1": True, "2": False},
            "1": {"5": True, "6": True},
            "2": {"0": True, "3": True}
        }
    }
    
    brain_result = test_api_endpoint(
        "/brain/process",
        method="POST",
        data=pattern_data,
        description="Neural pattern processing"
    )
    
    # Test 4: Text to pattern conversion
    text_result = test_api_endpoint(
        "/brain/text-to-pattern?text=Hello brain, how are you processing this neural input?",
        description="Text to neural pattern conversion"
    )
    
    # Test 5: Chat interaction
    chat_data = {
        "message": "Hello! Can you tell me about your neural network architecture?",
        "user_id": "test_user_123"
    }
    
    chat_result = test_api_endpoint(
        "/chat",
        method="POST", 
        data=chat_data,
        description="Interactive chat with brain"
    )
    
    # Test 6: Another chat to test conversation memory
    chat_data2 = {
        "message": "What can you tell me about artificial intelligence and neural networks?",
        "user_id": "test_user_123" 
    }
    
    chat_result2 = test_api_endpoint(
        "/chat",
        method="POST",
        data=chat_data2, 
        description="Follow-up chat message"
    )
    
    # Test 7: Brain state
    state_result = test_api_endpoint(
        "/brain/state",
        description="Current brain state"
    )
    
    # Test 8: Conversation history
    conv_result = test_api_endpoint(
        "/brain/conversations", 
        description="Conversation history"
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    tests = [
        ("Root endpoint", root_result),
        ("Health check", health_result), 
        ("Neural processing", brain_result),
        ("Text conversion", text_result),
        ("Chat interaction", chat_result),
        ("Follow-up chat", chat_result2),
        ("Brain state", state_result),
        ("Conversations", conv_result)
    ]
    
    passed = sum(1 for _, result in tests if result is not None)
    total = len(tests)
    
    print(f"✅ Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests successful! Your Brain API is working perfectly!")
        
        # Show some interesting results
        if chat_result:
            print(f"\n💬 Brain Response: {chat_result['response_text']}")
            print(f"🧠 Confidence: {chat_result['confidence_score']:.2f}")
            print(f"⚡ Processing time: {chat_result['processing_time_ms']:.1f}ms")
        
        if state_result:
            brain_data = state_result['data']
            print(f"\n🧠 Brain Status: {brain_data['status']}")
            print(f"🔗 Total neurons: {brain_data['total_neurons']}")
            print(f"📚 Conversations: {brain_data.get('conversation_history', 0)}")
    else:
        print("⚠️  Some tests failed. Check API server status.")

def start_server_instructions():
    """Show instructions for starting the API server"""
    print("🚀 To start the API server, run:")
    print(f"cd {'/'.join(__file__.split('/')[:-2])}")
    print("uvicorn src.api.simple_api:app --host 0.0.0.0 --port 8000 --reload")
    print("\nThen run this test script again!")

if __name__ == "__main__":
    # Quick connection test
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        main()
    except requests.exceptions.ConnectionError:
        print("🔌 API server not running!")
        start_server_instructions()
    except Exception as e:
        print(f"❌ Connection error: {e}")
        start_server_instructions()
