#!/usr/bin/env python3
"""
Test script to verify Qdrant vector database connection
"""

import sys
import time
from src.clients import api_clients

def test_qdrant_connection():
    """Test Qdrant connection and basic functionality"""
    print("Testing Qdrant Vector Database connection...")
    print("=" * 50)
    
    try:
        # Test basic connection
        start_time = time.time()
        result = api_clients.test_qdrant_connection()
        end_time = time.time()
        
        if result:
            print("✅ Qdrant connection: SUCCESSFUL")
            print(f"   Connection time: {end_time - start_time:.2f} seconds")
        else:
            print("❌ Qdrant connection: FAILED")
            return False
        
    except Exception as e:
        print(f"❌ Qdrant test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_qdrant_connection()
    sys.exit(0 if success else 1)