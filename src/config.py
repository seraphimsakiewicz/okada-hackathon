import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    environment: str = "development"
    debug: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def validate_required_keys(self) -> None:
        """Validate that all required API keys are present"""
        if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if not self.qdrant_url or self.qdrant_url == "your_qdrant_cloud_url_here":
            raise ValueError("QDRANT_URL environment variable is required")
        
        if not self.qdrant_api_key or self.qdrant_api_key == "your_qdrant_api_key_here":
            raise ValueError("QDRANT_API_KEY environment variable is required")

def get_settings() -> Settings:
    """Get application settings with validation"""
    settings = Settings()
    settings.validate_required_keys()
    return settings