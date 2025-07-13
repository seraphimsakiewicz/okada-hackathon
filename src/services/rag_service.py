import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, CollectionInfo, Filter, FieldCondition, MatchValue, Range
from qdrant_client.http.exceptions import UnexpectedResponse
import uuid
import time
import re
from ..clients import api_clients
from ..config import get_settings

logger = logging.getLogger(__name__)

class RAGService:
    """Handle RAG operations including embeddings and vector storage"""
    
    def __init__(self):
        self.settings = get_settings()
        self.collection_name = getattr(self.settings, 'qdrant_collection_name', 'voice_ai_documents')
        self.embedding_model = getattr(self.settings, 'embedding_model', 'text-embedding-3-small')
        self.embedding_dim = 1536  # OpenAI text-embedding-3-small dimensions
        
    def ensure_collection_exists(self) -> bool:
        """Ensure the Qdrant collection exists with proper configuration"""
        try:
            # Check if collection exists
            collections = api_clients.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                api_clients.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            return False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using OpenAI"""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            start_time = time.time()
            
            response = api_clients.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            end_time = time.time()
            
            logger.info(f"Generated {len(embeddings)} embeddings in {end_time - start_time:.2f}s")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index documents in the vector database"""
        try:
            if not documents:
                return {"indexed": 0, "message": "No documents to index"}
            
            # Ensure collection exists
            if not self.ensure_collection_exists():
                raise Exception("Failed to ensure collection exists")
            
            # Extract text content for embedding
            texts = [doc['content'] for doc in documents]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Prepare points for Qdrant
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point_id = str(uuid.uuid4())
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        'content': doc['content'],
                        'source': doc['source'],
                        'type': doc['type'],
                        'metadata': doc['metadata'],
                        'indexed_at': time.time()
                    }
                )
                points.append(point)
            
            # Upload to Qdrant
            start_time = time.time()
            operation_info = api_clients.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            end_time = time.time()
            
            logger.info(f"Indexed {len(points)} documents in {end_time - start_time:.2f}s")
            
            return {
                "indexed": len(points),
                "collection": self.collection_name,
                "indexing_time": end_time - start_time,
                "operation_info": operation_info.dict() if hasattr(operation_info, 'dict') else str(operation_info)
            }
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise
    
    def search_documents(self, query: str, limit: int = 5, score_threshold: float = 0.3, 
                        source_filter: Optional[str] = None, 
                        document_type_filter: Optional[str] = None,
                        uploaded_after: Optional[float] = None,
                        uploaded_before: Optional[float] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity with enhanced relevance scoring"""
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embeddings([query])[0]
            
            # Build filter conditions if specified
            filter_conditions = None
            if source_filter or document_type_filter or uploaded_after or uploaded_before:
                must_conditions = []
                
                if source_filter:
                    must_conditions.append(
                        FieldCondition(key="source", match=MatchValue(value=source_filter))
                    )
                if document_type_filter:
                    must_conditions.append(
                        FieldCondition(key="type", match=MatchValue(value=document_type_filter))
                    )
                
                # Add date range filtering
                if uploaded_after or uploaded_before:
                    range_condition = {}
                    if uploaded_after:
                        range_condition['gte'] = uploaded_after
                    if uploaded_before:
                        range_condition['lte'] = uploaded_before
                    
                    must_conditions.append(
                        FieldCondition(key="indexed_at", range=Range(**range_condition))
                    )
                
                if must_conditions:
                    filter_conditions = Filter(must=must_conditions)
            
            # Search in Qdrant with enhanced parameters
            start_time = time.time()
            
            # If we have filters that might not be indexed, get more results and filter afterward
            search_limit = limit * 3 if filter_conditions else limit
            
            try:
                search_results = api_clients.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=search_limit,
                    score_threshold=score_threshold,
                    query_filter=filter_conditions
                )
            except Exception as filter_error:
                if "Index required" in str(filter_error):
                    # Fallback: search without filters and post-filter
                    logger.warning(f"Filter not indexed, using post-filtering: {filter_error}")
                    search_results = api_clients.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding,
                        limit=search_limit,
                        score_threshold=score_threshold
                    )
                else:
                    raise filter_error
            end_time = time.time()
            
            # Enhanced result formatting with relevance scoring and post-filtering
            results = []
            for i, result in enumerate(search_results):
                # Apply post-filtering if needed
                if source_filter and source_filter not in result.payload.get('source', ''):
                    continue
                if document_type_filter and document_type_filter != result.payload.get('type'):
                    continue
                if uploaded_after and result.payload.get('indexed_at', 0) < uploaded_after:
                    continue
                if uploaded_before and result.payload.get('indexed_at', float('inf')) > uploaded_before:
                    continue
                
                # Calculate enhanced relevance score (combining vector similarity with ranking position)
                position_boost = 1.0 - (len(results) * 0.1)  # Slightly boost higher-ranked results
                enhanced_score = result.score * position_boost
                
                results.append({
                    'content': result.payload['content'],
                    'source': result.payload['source'],
                    'type': result.payload['type'],
                    'metadata': result.payload['metadata'],
                    'score': result.score,
                    'enhanced_score': enhanced_score,
                    'rank': len(results) + 1,
                    'id': result.id,
                    'indexed_at': result.payload.get('indexed_at')
                })
                
                # Stop when we have enough results
                if len(results) >= limit:
                    break
            
            # Sort by enhanced score for final ranking
            results.sort(key=lambda x: x['enhanced_score'], reverse=True)
            
            logger.info(f"Found {len(results)} results for query '{query}' in {end_time - start_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    async def resolve_references_with_llm(self, query: str, conversation_history: List[Dict]) -> str:
        """Use GPT to rewrite queries with resolved references"""
        try:
            # Check if query contains references that need resolution
            reference_keywords = ["that", "it", "this", "the property", "that property", "that place", "above", "previous"]
            if not any(keyword in query.lower() for keyword in reference_keywords):
                logger.info(f"🔍 No references detected in query: '{query}' - skipping resolution")
                return query  # No references to resolve
            
            logger.info(f"🔄 Resolving references in query: '{query}'")
            
            # Get recent conversation context (last 2-3 exchanges)
            recent_context = []
            for msg in conversation_history[-6:]:  # Last 3 exchanges (user + assistant)
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if content.strip():
                    recent_context.append(f"{role.capitalize()}: {content[:500]}")  # Limit length
            
            context_text = "\n".join(recent_context) if recent_context else "No previous conversation context available."
            
            # Create rewrite prompt
            rewrite_prompt = f"""Given this conversation context:
{context_text}

Rewrite the following query to be explicit by replacing vague references like "that property", "it", "this", "that place" with specific addresses, property details, or concrete terms mentioned in the conversation context.

Query to rewrite: {query}

Rules:
- If the context mentions specific addresses, use them to replace references
- If no specific address is clear, keep the query as is
- Return ONLY the rewritten query, nothing else
- Do not add extra words or explanations

Rewritten query:"""

            logger.info(f"📝 Sending rewrite prompt to LLM...")
            
            # Use OpenAI to rewrite the query
            response = api_clients.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": rewrite_prompt}],
                max_tokens=100,
                temperature=0.1  # Low temperature for consistent rewrites
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            
            # Log the rewrite result with additional debug info
            if rewritten_query.lower() != query.lower():
                logger.info(f"✅ FOLLOW-UP RESOLVED: '{query}' → '{rewritten_query}'")
            else:
                logger.info(f"🔄 No changes needed: '{query}'")
                
            return rewritten_query
            
        except Exception as e:
            logger.error(f"❌ Error resolving references with LLM: {e}")
            # Fallback to original query if rewriting fails
            return query

    async def get_top_relevant_chunks(self, query: str, k: int = 3, conversation_history: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Get top-k most relevant document chunks optimized for LLM context injection with reference resolution"""
        try:
            # Ensure k is within reasonable bounds (3-5 for optimal context)
            k = max(3, min(k, 5))
            
            # Resolve references if conversation history is provided
            resolved_query = query
            if conversation_history:
                logger.info(f"🗣️ Conversation history provided: {len(conversation_history)} messages")
                # Use await since we're in an async function
                try:
                    resolved_query = await self.resolve_references_with_llm(query, conversation_history)
                except Exception as resolve_error:
                    logger.warning(f"⚠️ Reference resolution failed, using original query: {resolve_error}")
                    resolved_query = query
            else:
                logger.info(f"🚫 No conversation history provided for reference resolution")
            
            # Check if this is an address-specific query (use resolved query)
            address_pattern = r'\d+\s+[NESW]\s+\d+\w*\s+[Ss]t(?:reet)?'
            address_match = re.search(address_pattern, resolved_query, re.IGNORECASE)
            
            if address_match:
                logger.info(f"🏠 Detected address query: {address_match.group()}")
                # Use hybrid search for address queries
                return self._hybrid_address_search(resolved_query, address_match.group(), k)
            
            # Use moderate score threshold for top-k to ensure quality
            score_threshold = 0.2  # Lower threshold to get more candidates
            
            # Search with higher limit to have more candidates for filtering (use resolved query)
            results = self.search_documents(
                query=resolved_query,  # Use the resolved query for better results
                limit=k * 2,  # Get more candidates
                score_threshold=score_threshold
            )
            
            # Filter to top-k results with additional quality checks
            top_results = []
            for result in results[:k]:
                # Additional quality filtering
                content_length = len(result['content'])
                if content_length > 50:  # Ensure minimum content length
                    top_results.append(result)
            
            # If we don't have enough quality results, get more with lower threshold
            if len(top_results) < k:
                additional_results = self.search_documents(
                    query=resolved_query,  # Use resolved query here too
                    limit=k * 3,
                    score_threshold=0.1  # Much lower threshold for fallback
                )
                
                for result in additional_results:
                    if len(top_results) >= k:
                        break
                    if result not in top_results and len(result['content']) > 50:
                        top_results.append(result)
            
            logger.info(f"📄 Retrieved {len(top_results)} chunks for query: '{query}' → resolved: '{resolved_query}'")
            if top_results:
                logger.info(f"🏆 Top result score: {top_results[0].get('enhanced_score', top_results[0].get('score', 'N/A')):.3f}")
            return top_results[:k]  # Ensure we return exactly k results
            
        except Exception as e:
            logger.error(f"Error getting top relevant chunks: {e}")
            raise
    
    def _hybrid_address_search(self, query: str, address: str, k: int = 3) -> List[Dict[str, Any]]:
        """Hybrid search that prioritizes exact address matches"""
        try:
            logger.info(f"🔍 Hybrid address search for: '{address}'")
            
            # Get semantic search results first
            all_results = self.search_documents(
                query=address,  # Search for the address directly, not the full query
                limit=20,  # Get more candidates
                score_threshold=0.1
            )
            
            exact_matches = []
            partial_matches = []
            
            # Normalize the search address for comparison (handle different formats)
            search_address = address.strip().lower().replace("street", "st").replace(".", "")
            logger.info(f"🏠 Normalized search address: '{search_address}'")
            
            for result in all_results:
                content = result['content'].lower()
                
                # Extract the property address from content using regex
                import re
                address_in_content = re.search(r'property address:\s*([^|]+)', content)
                if address_in_content:
                    found_address = address_in_content.group(1).strip().lower().replace("street", "st").replace(".", "")
                    logger.info(f"📍 Comparing '{search_address}' with '{found_address}'")
                    
                    if search_address == found_address:
                        exact_matches.append(result)
                        logger.info(f"✅ EXACT match found: {found_address}")
                    elif search_address in found_address or found_address in search_address:
                        partial_matches.append(result)
                        logger.info(f"🟡 Partial match: {found_address}")
            
            # Prioritize exact matches, then partial matches, then semantic matches
            remaining_semantic = [r for r in all_results if r not in exact_matches and r not in partial_matches]
            combined_results = exact_matches + partial_matches + remaining_semantic
            
            logger.info(f"📊 Found {len(exact_matches)} exact, {len(partial_matches)} partial, {len(remaining_semantic)} semantic matches")
            
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid address search: {e}")
            return self.search_documents(query=query, limit=k, score_threshold=0.2)
    
    def format_context_for_llm(self, retrieved_docs: List[Dict[str, Any]], query: str) -> str:
        """Format retrieved documents into context string optimized for LLM injection"""
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = [
            "Based on the following relevant information from the knowledge base:",
            ""
        ]
        
        for i, doc in enumerate(retrieved_docs, 1):
            # Extract key information
            content = doc['content']
            source = doc['source']
            score = doc.get('enhanced_score', doc.get('score', 0))
            
            # Format each document chunk
            doc_header = f"[Document {i} - Source: {source} - Relevance: {score:.3f}]"
            context_parts.append(doc_header)
            context_parts.append(content)
            context_parts.append("")  # Add spacing between documents
        
        context_parts.extend([
            "Please use this information to provide accurate and helpful responses about real estate properties, pricing, and availability.",
            "If the query cannot be answered using the provided context, please state that clearly.",
            ""
        ])
        
        formatted_context = "\n".join(context_parts)
        
        # Log context generation for debugging
        logger.info(f"Generated context for query '{query}' with {len(retrieved_docs)} documents ({len(formatted_context)} chars)")
        
        return formatted_context
    
    async def get_contextualized_response_data(self, query: str, k: int = 3, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Get both retrieved documents and formatted context for LLM in one call with reference resolution"""
        try:
            # Get top relevant chunks with conversation history for reference resolution
            relevant_docs = await self.get_top_relevant_chunks(query, k, conversation_history)
            
            # Format context for LLM
            formatted_context = self.format_context_for_llm(relevant_docs, query)
            
            return {
                'query': query,
                'retrieved_documents': relevant_docs,
                'formatted_context': formatted_context,
                'document_count': len(relevant_docs),
                'context_length': len(formatted_context)
            }
            
        except Exception as e:
            logger.error(f"Error getting contextualized response data: {e}")
            raise
    
    def search_by_metadata(self, query: str, 
                          source_contains: Optional[str] = None,
                          document_type: Optional[str] = None, 
                          last_n_days: Optional[int] = None,
                          limit: int = 5) -> List[Dict[str, Any]]:
        """Convenience method for searching with common metadata filters"""
        try:
            # Calculate date filter if specified
            uploaded_after = None
            if last_n_days:
                uploaded_after = time.time() - (last_n_days * 24 * 60 * 60)
            
            # Apply partial source matching
            source_filter = None
            if source_contains:
                # Note: This is a simplified approach. For exact matching, we'd need to implement
                # more sophisticated filtering in the search_documents method
                source_filter = source_contains
            
            results = self.search_documents(
                query=query,
                limit=limit,
                source_filter=source_filter,
                document_type_filter=document_type,
                uploaded_after=uploaded_after
            )
            
            # Additional post-processing for partial source matching
            if source_contains and not source_filter:
                results = [r for r in results if source_contains.lower() in r['source'].lower()]
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            raise
    
    def get_document_sources(self) -> List[str]:
        """Get list of all unique document sources in the collection"""
        try:
            # Use scroll API to get all points efficiently
            scroll_result = api_clients.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False  # We don't need vectors, just payload
            )
            
            sources = set()
            for point in scroll_result[0]:  # scroll returns (points, next_page_offset)
                if hasattr(point, 'payload') and point.payload:
                    source = point.payload.get('source', 'unknown')
                    if source != 'unknown':
                        sources.add(source)
            
            return sorted(list(sources))
            
        except Exception as e:
            logger.error(f"Error getting document sources: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            # Use a simpler approach to get collection info
            collections = api_clients.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                return {
                    "name": self.collection_name,
                    "exists": False,
                    "points_count": 0
                }
            
            # Get points count using count API
            count_result = api_clients.qdrant_client.count(
                collection_name=self.collection_name
            )
            
            return {
                "name": self.collection_name,
                "exists": True,
                "points_count": count_result.count if hasattr(count_result, 'count') else 0,
                "vector_size": self.embedding_dim,
                "distance": "cosine"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the current collection (for testing/reset purposes)"""
        try:
            api_clients.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False