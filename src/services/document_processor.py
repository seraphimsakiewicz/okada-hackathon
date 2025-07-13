import os
import json
import pandas as pd
from typing import List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handle document text extraction and processing"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.txt', '.csv', '.json'}
    
    def extract_text(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract text from uploaded file based on format"""
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Extracting text from {filename} ({file_ext})")
        
        if file_ext == '.txt':
            return self._extract_from_txt(file_path, filename)
        elif file_ext == '.csv':
            return self._extract_from_csv(file_path, filename)
        elif file_ext == '.json':
            return self._extract_from_json(file_path, filename)
        elif file_ext == '.pdf':
            return self._extract_from_pdf(file_path, filename)
        else:
            raise ValueError(f"Extraction not implemented for {file_ext}")
    
    def _extract_from_txt(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            return [{
                'content': content,
                'source': filename,
                'type': 'text',
                'metadata': {
                    'filename': filename,
                    'file_type': 'txt',
                    'size': len(content)
                }
            }]
        except Exception as e:
            logger.error(f"Error extracting from TXT {filename}: {e}")
            raise
    
    def _extract_from_csv(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            documents = []
            
            for index, row in df.iterrows():
                # Convert row to string representation
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                
                documents.append({
                    'content': row_text,
                    'source': filename,
                    'type': 'csv_row',
                    'metadata': {
                        'filename': filename,
                        'file_type': 'csv',
                        'row_index': index,
                        'columns': list(df.columns)
                    }
                })
            
            logger.info(f"Extracted {len(documents)} rows from CSV {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting from CSV {filename}: {e}")
            raise
    
    def _extract_from_json(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract text from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            documents = []
            
            if isinstance(data, list):
                # List of objects
                for index, item in enumerate(data):
                    content = json.dumps(item, indent=2) if isinstance(item, (dict, list)) else str(item)
                    documents.append({
                        'content': content,
                        'source': filename,
                        'type': 'json_item',
                        'metadata': {
                            'filename': filename,
                            'file_type': 'json',
                            'item_index': index
                        }
                    })
            elif isinstance(data, dict):
                # Single object or nested structure
                content = json.dumps(data, indent=2)
                documents.append({
                    'content': content,
                    'source': filename,
                    'type': 'json_object',
                    'metadata': {
                        'filename': filename,
                        'file_type': 'json',
                        'keys': list(data.keys()) if isinstance(data, dict) else None
                    }
                })
            else:
                # Primitive value
                documents.append({
                    'content': str(data),
                    'source': filename,
                    'type': 'json_primitive',
                    'metadata': {
                        'filename': filename,
                        'file_type': 'json',
                        'data_type': type(data).__name__
                    }
                })
            
            logger.info(f"Extracted {len(documents)} items from JSON {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting from JSON {filename}: {e}")
            raise
    
    def _extract_from_pdf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            
            documents = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        documents.append({
                            'content': text,
                            'source': filename,
                            'type': 'pdf_page',
                            'metadata': {
                                'filename': filename,
                                'file_type': 'pdf',
                                'page_number': page_num + 1,
                                'total_pages': len(pdf_reader.pages)
                            }
                        })
            
            logger.info(f"Extracted {len(documents)} pages from PDF {filename}")
            return documents
            
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise ImportError("PyPDF2 required for PDF processing")
        except Exception as e:
            logger.error(f"Error extracting from PDF {filename}: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If we're not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence ending or paragraph break
                break_point = text.rfind('.', start, end)
                if break_point == -1:
                    break_point = text.rfind('\n', start, end)
                if break_point == -1:
                    break_point = text.rfind(' ', start, end)
                
                if break_point > start:
                    end = break_point + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap if end < len(text) else len(text)
        
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Process documents into chunks with metadata"""
        processed_chunks = []
        
        for doc in documents:
            content = doc['content']
            chunks = self.chunk_text(content, chunk_size, overlap)
            
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    'content': chunk,
                    'source': doc['source'],
                    'type': doc['type'],
                    'metadata': {
                        **doc['metadata'],
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk)
                    }
                }
                processed_chunks.append(chunk_doc)
        
        logger.info(f"Created {len(processed_chunks)} chunks from {len(documents)} documents")
        return processed_chunks