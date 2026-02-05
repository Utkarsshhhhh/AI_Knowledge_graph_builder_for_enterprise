"""
Configuration Module for RAG System
Centralized configuration management with environment variable support
"""

import os
from pathlib import Path

# ============================================================================
# BASE PATHS
# ============================================================================

# Automatically detect the project root directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_FOLDER = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Output"

# Ensure directories exist
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILE PATHS
# ============================================================================

# Ingested data (from Data_Ingestion.py)
INGESTED_DATA_PATH = BASE_DIR / "ingested_data.json"

# Extracted entities (from Entity_Extraction.py)
EXTRACTED_ENTITIES_PATH = BASE_DIR / "extracted_entities_clean.json"

# Extracted relationships (from Relationship_Ext.py)
EXTRACTED_RELATIONSHIPS_PATH = BASE_DIR / "extracted_relationships_clean.json"

# Knowledge triples (from Triple_Crt.py)
# Using standardized schema: subject, predicate, object
KNOWLEDGE_TRIPLES_PATH = BASE_DIR / "entity_relation_entity_triples.json"
KNOWLEDGE_TRIPLES_TXT_PATH = BASE_DIR / "entity_relation_entity_triples.txt"

# Vector database files (from Embedding_gen.py)
VECTOR_INDEX_PATH = BASE_DIR / "vector_database.index"
VECTOR_METADATA_PATH = BASE_DIR / "vector_metadata.json"

# ============================================================================
# NEO4J CONFIGURATION
# ============================================================================

NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "Anand@1234"),  # Change in production!
    "database": os.getenv("NEO4J_DATABASE", "neo4j")
}

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

EMBEDDING_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    "dimension": 384,  # Dimension for all-MiniLM-L6-v2
    "top_k": int(os.getenv("TOP_K", "5")),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50"))
}

# ============================================================================
# LLM CONFIGURATION (Ollama)
# ============================================================================

OLLAMA_CONFIG = {
    "enabled": os.getenv("USE_LLM", "false").lower() == "true",
    "api_url": os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
    "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
    "timeout": int(os.getenv("OLLAMA_TIMEOUT", "30")),
    "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
    "max_tokens": int(os.getenv("OLLAMA_MAX_TOKENS", "200"))
}

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

APP_CONFIG = {
    "host": os.getenv("FLASK_HOST", "0.0.0.0"),
    "port": int(os.getenv("FLASK_PORT", "5000")),
    "debug": os.getenv("FLASK_DEBUG", "false").lower() == "true",
    "max_query_length": int(os.getenv("MAX_QUERY_LENGTH", "500"))
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": BASE_DIR / "rag_system.log"
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config():
    """
    Validate that all required files and configurations are present
    
    Returns:
        dict: Validation results with status and messages
    """
    results = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check required data files
    required_files = {
        "Vector Index": VECTOR_INDEX_PATH,
        "Vector Metadata": VECTOR_METADATA_PATH,
        "Knowledge Triples": KNOWLEDGE_TRIPLES_PATH
    }
    
    for name, path in required_files.items():
        if not path.exists():
            results["errors"].append(f"{name} not found: {path}")
            results["valid"] = False
    
    # Check optional files (warnings only)
    optional_files = {
        "Ingested Data": INGESTED_DATA_PATH,
        "Extracted Entities": EXTRACTED_ENTITIES_PATH,
        "Extracted Relationships": EXTRACTED_RELATIONSHIPS_PATH
    }
    
    for name, path in optional_files.items():
        if not path.exists():
            results["warnings"].append(f"{name} not found: {path}")
    
    # Check Neo4j password
    if NEO4J_CONFIG["password"] == "Anand@1234":
        results["warnings"].append("Using default Neo4j password. Change in production!")
    
    return results

def print_config_summary():
    """Print configuration summary for debugging"""
    print("=" * 70)
    print("RAG SYSTEM CONFIGURATION")
    print("=" * 70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Folder: {DATA_FOLDER}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print()
    print("Neo4j:")
    print(f"  URI: {NEO4J_CONFIG['uri']}")
    print(f"  User: {NEO4J_CONFIG['user']}")
    print(f"  Database: {NEO4J_CONFIG['database']}")
    print()
    print("Embedding:")
    print(f"  Model: {EMBEDDING_CONFIG['model_name']}")
    print(f"  Top-K: {EMBEDDING_CONFIG['top_k']}")
    print(f"  Chunk Size: {EMBEDDING_CONFIG['chunk_size']}")
    print()
    print("Ollama LLM:")
    print(f"  Enabled: {OLLAMA_CONFIG['enabled']}")
    print(f"  Model: {OLLAMA_CONFIG['model']}")
    print(f"  URL: {OLLAMA_CONFIG['api_url']}")
    print()
    print("Application:")
    print(f"  Host: {APP_CONFIG['host']}")
    print(f"  Port: {APP_CONFIG['port']}")
    print(f"  Debug: {APP_CONFIG['debug']}")
    print("=" * 70)

# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    print_config_summary()
    
    validation = validate_config()
    
    print("\nValidation Results:")
    print("-" * 70)
    
    if validation["errors"]:
        print("\n❌ ERRORS:")
        for error in validation["errors"]:
            print(f"  • {error}")
    
    if validation["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in validation["warnings"]:
            print(f"  • {warning}")
    
    if validation["valid"]:
        print("\n✅ Configuration is valid!")
    else:
        print("\n❌ Configuration has errors. Please fix them before running the system.")