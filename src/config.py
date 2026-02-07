"""
Configuration settings for the RF Communications Expert System
"""
import os
from pathlib import Path

# Application settings
APP_TITLE = "RF Communications Expert System"
APP_DESCRIPTION = "AI-powered expert system for RF communications queries"

# File paths
DATA_DIR = Path("data")
VECTOR_DB_DIR = DATA_DIR / "vector_db"
DOCUMENTS_DIR = DATA_DIR / "documents"
SCRAPED_DIR = DATA_DIR / "scraped"
AUDIT_LOGS_DIR = Path(os.getenv("AUDIT_LOG_PATH", "audit_logs"))

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)
SCRAPED_DIR.mkdir(exist_ok=True)
AUDIT_LOGS_DIR.mkdir(exist_ok=True)

# LLM Backend settings
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama2")

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# LocalAI settings
LOCALAI_HOST = os.getenv("LOCALAI_HOST", "http://localhost:8080")
LOCALAI_MODEL = os.getenv("LOCALAI_MODEL", "ggml-gpt4all-j")

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Audit and security settings
ENABLE_AUDIT_LOGGING = os.getenv("ENABLE_AUDIT_LOGGING", "false").lower() == "true"
ENABLE_DLP = os.getenv("ENABLE_DLP", "false").lower() == "true"
DATA_CLASSIFICATION = os.getenv("DATA_CLASSIFICATION", "RESTRICTED")
COMPLIANCE_FRAMEWORK = os.getenv("COMPLIANCE_FRAMEWORK", "SOC2")

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.7

# Supported file types
SUPPORTED_FILE_TYPES = ['.pdf', '.docx', '.doc', '.txt']

# Default RF communications websites for scraping
DEFAULT_RF_WEBSITES = [
    "https://www.rfcafe.com",
    "https://www.microwaves101.com",
    "https://www.everythingrf.com",
    "https://www.antenna-theory.com"
]

# System prompts
RF_EXPERT_PROMPT = """You are an expert in RF (Radio Frequency) communications with deep knowledge in:
- Antenna theory and design
- RF circuit design and analysis
- Microwave engineering
- Signal propagation
- RF test and measurement
- Wireless communication systems
- EMI/EMC considerations
- RF safety and regulations

Please provide detailed, technical responses based on the context provided. If you're unsure about something, indicate your uncertainty. Always prioritize accuracy and safety in RF applications.

Context: {context}

Question: {question}

Answer:"""
