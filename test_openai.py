#!/usr/bin/env python3
"""
Test script to verify OpenAI API connection
"""

import sys
import time
from src.clients import api_clients

def test_openai_connection():
    """Test OpenAI API connection and basic functionality"""
    print("Testing OpenAI API connection...")
    print("=" * 50)
    
    try:
        # Test basic connection
        start_time = time.time()
        result = api_clients.test_openai_connection()
        end_time = time.time()
        
        if result:
            print("✅ OpenAI connection: SUCCESSFUL")
            print(f"   Connection time: {end_time - start_time:.2f} seconds")
        else:
            print("❌ OpenAI connection: FAILED")
            return False
        
        # Test model listing
        print("\n📋 Available models:")
        models = api_clients.openai_client.models.list()
        model_count = 0
        for model in models.data[:5]:  # Show first 5 models
            print(f"   - {model.id}")
            model_count += 1
        print(f"   ... and {len(models.data) - 5} more models")
        
        # Test basic completion
        print("\n🤖 Testing text completion:")
        start_time = time.time()
        response = api_clients.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello, OpenAI API is working!'"}],
            max_tokens=50
        )
        end_time = time.time()
        
        print(f"   Response: {response.choices[0].message.content}")
        print(f"   Generation time: {end_time - start_time:.2f} seconds")
        print(f"   Tokens used: {response.usage.total_tokens}")
        
        # Test embedding
        print("\n🔢 Testing text embedding:")
        start_time = time.time()
        embedding_response = api_clients.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input="This is a test embedding"
        )
        end_time = time.time()
        
        embedding = embedding_response.data[0].embedding
        print(f"   Embedding dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Embedding time: {end_time - start_time:.2f} seconds")
        
        print("\n🎉 All OpenAI tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openai_connection()
    sys.exit(0 if success else 1)