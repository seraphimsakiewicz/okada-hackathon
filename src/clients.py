import openai
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from typing import Optional
import logging
from .config import get_settings

logger = logging.getLogger(__name__)

class APIClients:
    def __init__(self):
        self.settings = get_settings()
        self._openai_client: Optional[openai.OpenAI] = None
        self._qdrant_client: Optional[QdrantClient] = None
    
    @property
    def openai_client(self) -> openai.OpenAI:
        """Get OpenAI client instance with lazy initialization"""
        if self._openai_client is None:
            try:
                self._openai_client = openai.OpenAI(
                    api_key=self.settings.openai_api_key
                )
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise ConnectionError(f"OpenAI client initialization failed: {e}")
        return self._openai_client
    
    @property
    def qdrant_client(self) -> QdrantClient:
        """Get Qdrant client instance with lazy initialization"""
        if self._qdrant_client is None:
            try:
                self._qdrant_client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key
                )
                logger.info("Qdrant client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant client: {e}")
                raise ConnectionError(f"Qdrant client initialization failed: {e}")
        return self._qdrant_client
    
    def test_openai_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            response = self.openai_client.models.list()
            logger.info("OpenAI connection test successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
    
    def test_qdrant_connection(self) -> bool:
        """Test Qdrant connection"""
        try:
            collections = self.qdrant_client.get_collections()
            logger.info("Qdrant connection test successful")
            return True
        except Exception as e:
            logger.error(f"Qdrant connection test failed: {e}")
            return False

# Global instance
api_clients = APIClients()