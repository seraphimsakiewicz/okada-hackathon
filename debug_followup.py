#!/usr/bin/env python3
"""Debug script for follow-up query resolution"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.rag_service import RAGService
from src.clients import api_clients
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_followup_resolution():
    """Test follow-up query resolution"""
    rag_service = RAGService()
    
    # Simulate conversation history
    conversation_history = [
        {"role": "user", "content": "Who are the associates that manage the property on 36 W 36th street?"},
        {"role": "assistant", "content": "The associates managing the property at 36 W 36th St are:\n\n1. Hector Barbossa\n2. Jorah Mormont\n3. Meemaw\n4. Oscar Piastri\n\nIf you need more information about the property or the associates, feel free to ask!"}
    ]
    
    # Test query with reference
    follow_up_query = "What is the rent for that property?"
    
    logger.info(f"=== TESTING FOLLOW-UP RESOLUTION ===")
    logger.info(f"Original query: '{follow_up_query}'")
    logger.info(f"Conversation history: {len(conversation_history)} messages")
    
    # Test reference resolution
    resolved_query = await rag_service.resolve_references_with_llm(follow_up_query, conversation_history)
    logger.info(f"Resolved query: '{resolved_query}'")
    
    # Test RAG retrieval with resolved query
    logger.info(f"\n=== TESTING RAG RETRIEVAL ===")
    
    # Test original query
    original_results = rag_service.get_contextualized_response_data(
        query=follow_up_query,
        k=3
    )
    logger.info(f"Original query results: {len(original_results.get('retrieved_documents', []))} docs")
    if original_results.get('retrieved_documents'):
        logger.info(f"Top result score: {original_results['retrieved_documents'][0].get('score', 'N/A')}")
    
    # Test resolved query
    resolved_results = rag_service.get_contextualized_response_data(
        query=resolved_query,
        k=3
    )
    logger.info(f"Resolved query results: {len(resolved_results.get('retrieved_documents', []))} docs")
    if resolved_results.get('retrieved_documents'):
        logger.info(f"Top result score: {resolved_results['retrieved_documents'][0].get('score', 'N/A')}")
        logger.info(f"Top result content preview: {resolved_results['retrieved_documents'][0].get('content', '')[:200]}...")
    
    # Test direct address search
    logger.info(f"\n=== TESTING DIRECT ADDRESS SEARCH ===")
    direct_results = rag_service.get_contextualized_response_data(
        query="What is the rent for 36 W 36th St?",
        k=3
    )
    logger.info(f"Direct address results: {len(direct_results.get('retrieved_documents', []))} docs")
    if direct_results.get('retrieved_documents'):
        logger.info(f"Top result score: {direct_results['retrieved_documents'][0].get('score', 'N/A')}")
        logger.info(f"Top result content preview: {direct_results['retrieved_documents'][0].get('content', '')[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_followup_resolution())