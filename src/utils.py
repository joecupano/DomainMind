"""
Utility functions for the RF Communications Expert System
"""
import re
import logging
from typing import List, Optional
from pathlib import Path
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\{\}]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def validate_file_type(file_path: Path) -> bool:
    """Validate if file type is supported"""
    from config import SUPPORTED_FILE_TYPES
    return file_path.suffix.lower() in SUPPORTED_FILE_TYPES

def display_error(error_message: str):
    """Display error message in Streamlit"""
    st.error(f"❌ Error: {error_message}")
    logger.error(error_message)

def display_success(success_message: str):
    """Display success message in Streamlit"""
    st.success(f"✅ {success_message}")
    logger.info(success_message)

def display_info(info_message: str):
    """Display info message in Streamlit"""
    st.info(f"ℹ️ {info_message}")
    logger.info(info_message)

def display_warning(warning_message: str):
    """Display warning message in Streamlit"""
    st.warning(f"⚠️ {warning_message}")
    logger.warning(warning_message)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def is_url_valid(url: str) -> bool:
    """Basic URL validation"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None
