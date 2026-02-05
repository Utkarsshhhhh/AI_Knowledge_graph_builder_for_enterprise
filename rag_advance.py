from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import time
import os
import sys
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from neo4j import GraphDatabase
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from Config import BASE_DIR, NEO4J_CONFIG, EMBEDDING_CONFIG
    USE_CONFIG = True
except ImportError:
    print("⚠️ Config.py not found, using defaults")
    BASE_DIR = Path(__file__).parent
    NEO4J_CONFIG = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": os.getenv("NEO4J_PASSWORD", "password"),
        "database": "neo4j"
    }
    EMBEDDING_CONFIG = {
        "model_name": "all-MiniLM-L6-v2",
        "top_k": 5
    }
    USE_CONFIG = False

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)

# ✅ SECURITY: Restrict CORS to specific origins
CORS(app, origins=[
    "http://localhost:5000",
    "http://127.0.0.1:5000"
])

# --------------------------------------------------
# Configuration
# --------------------------------------------------
VECTOR_INDEX = BASE_DIR / "vector_database.index"
VECTOR_METADATA = BASE_DIR / "vector_metadata.json"
KNOWLEDGE_TRIPLES = BASE_DIR / "entity_relation_entity_triples.json"

# Ollama configuration
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# --------------------------------------------------
# Globals
# --------------------------------------------------
model = None
faiss_index = None
chunks = []
triples = []
neo4j_driver = None

# --------------------------------------------------
# Logging Setup
# --------------------------------------------------
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Initialization
# --------------------------------------------------
def initialize():
    """Initialize all components with proper error handling"""
    global model, faiss_index, chunks, triples, neo4j_driver

    logger.info("=" * 60)
    logger.info("Initializing RAG Backend...")
    logger.info("=" * 60)

    # 1. Load embedding model
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_CONFIG['model_name']}")
        model = SentenceTransformer(EMBEDDING_CONFIG["model_name"])
        logger.info("✅ Embedding model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        raise

    # 2. Load FAISS index
    try:
        if not VECTOR_INDEX.exists():
            raise FileNotFoundError(f"FAISS index not found: {VECTOR_INDEX}")
        
        faiss_index = faiss.read_index(str(VECTOR_INDEX))
        logger.info(f"✅ Loaded FAISS index ({faiss_index.ntotal} vectors)")
    except Exception as e:
        logger.error(f"❌ Failed to load FAISS index: {e}")
        raise

    # 3. Load metadata
    try:
        if not VECTOR_METADATA.exists():
            raise FileNotFoundError(f"Metadata not found: {VECTOR_METADATA}")
        
        with open(VECTOR_METADATA, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            chunks = metadata.get("chunks", [])
        
        logger.info(f"✅ Loaded {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"❌ Failed to load metadata: {e}")
        raise

    # 4. Load triples
    try:
        if not KNOWLEDGE_TRIPLES.exists():
            logger.warning(f"⚠️ Triples file not found: {KNOWLEDGE_TRIPLES}")
            triples = []
        else:
            with open(KNOWLEDGE_TRIPLES, "r", encoding="utf-8") as f:
                triples = json.load(f)
            logger.info(f"✅ Loaded {len(triples)} triples")
    except Exception as e:
        logger.error(f"❌ Failed to load triples: {e}")
        triples = []

    # 5. Connect to Neo4j
    try:
        neo4j_driver = GraphDatabase.driver(
            NEO4J_CONFIG["uri"],
            auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"])
        )
        
        # Test connection
        with neo4j_driver.session(database=NEO4J_CONFIG["database"]) as session:
            session.run("RETURN 1").single()
        
        logger.info("✅ Connected to Neo4j")
    except Exception as e:
        logger.warning(f"⚠️ Neo4j connection failed: {e}")
        logger.warning("Graph search will be disabled")
        neo4j_driver = None

    logger.info("=" * 60)
    logger.info("✅ Initialization complete")
    logger.info("=" * 60)

    return {
        "chunks": len(chunks),
        "triples": len(triples),
        "neo4j": neo4j_driver is not None
    }

# --------------------------------------------------
# Vector Search
# --------------------------------------------------
def vector_search(query: str, top_k: int = 5):
    """
    Perform vector similarity search
    
    ✅ FIXED: Proper normalization
    """
    start = time.time()

    try:
        # Encode query
        q_vec = model.encode([query]).astype("float32")
        
        # ✅ CRITICAL: Normalize query vector
        faiss.normalize_L2(q_vec)

        # Search
        distances, indices = faiss_index.search(q_vec, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(chunks):
                continue

            c = chunks[idx]
            score = 1 / (1 + dist)

            results.append({
                "file": c.get("file", "unknown"),
                "chunk_id": c.get("chunk_id", 0),
                "content": c.get("content", ""),
                "score": round(score, 3)
            })

        latency = (time.time() - start) * 1000
        return results, round(latency, 2)
    
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return [], 0

# --------------------------------------------------
# Graph Search
# --------------------------------------------------
def graph_search(query: str, limit: int = 5):
    """
    Search knowledge graph
    
    ✅ FIXED: Backwards-compatible Neo4j syntax
    """
    start = time.time()
    
    if neo4j_driver is None:
        return [], 0

    try:
        with neo4j_driver.session(database=NEO4J_CONFIG["database"]) as session:
            # ✅ FIXED: Use size() instead of COUNT{} for compatibility
            cypher = """
            MATCH (n:Entity)
            WHERE toLower(n.name) CONTAINS toLower($q)
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n.name AS entity,
                   n.type AS type,
                   collect(DISTINCT {
                       name: m.name,
                       relation: type(r)
                   })[0..5] AS connections,
                   size((n)--()) AS degree
            ORDER BY degree DESC
            LIMIT $limit
            """

            records = session.run(cypher, q=query, limit=limit)

            results = []
            for r in records:
                results.append({
                    "entity": r["entity"],
                    "type": r["type"],
                    "degree": r["degree"],
                    "connections": [
                        c for c in r["connections"] 
                        if c and c.get("name")
                    ]
                })

        latency = (time.time() - start) * 1000
        return results, round(latency, 2)
    
    except Exception as e:
        logger.error(f"Graph search error: {e}")
        return [], 0

# --------------------------------------------------
# Triple Search
# --------------------------------------------------
def triple_search(query: str, limit: int = 5):
    """
    Search knowledge triples
    
    ✅ FIXED: Uses standardized schema (subject/predicate/object)
    """
    try:
        q = query.lower()
        results = []

        for t in triples:
            # ✅ STANDARDIZED: Use subject/predicate/object
            if (
                q in t.get("subject", "").lower()
                or q in t.get("predicate", "").lower()
                or q in t.get("object", "").lower()
            ):
                results.append({
                    "subject": t.get("subject", ""),
                    "predicate": t.get("predicate", ""),
                    "object": t.get("object", ""),
                    "source_file": t.get("source_file", "knowledge_graph"),
                    "confidence": t.get("confidence", 1.0)
                })

            if len(results) >= limit:
                break

        return results
    
    except Exception as e:
        logger.error(f"Triple search error: {e}")
        return []

# --------------------------------------------------
# Answer Generation (Template-based)
# --------------------------------------------------
def generate_answer_template(vec, graph, tri):
    """Generate answer from search results (no LLM)"""
    parts = []

    if vec:
        parts.append("📄 Document Evidence:")
        for r in vec[:3]:
            content = r['content'][:140]
            parts.append(f"• {content}... (from {r['file']})")

    if graph:
        parts.append("\n🕸️ Knowledge Graph Entities:")
        for g in graph[:3]:
            parts.append(f"• {g['entity']} ({g['type']}) - {g['degree']} connections")

    if tri:
        parts.append("\n🔗 Relationships:")
        for t in tri[:3]:
            parts.append(f"• {t['subject']} → {t['predicate']} → {t['object']}")

    return "\n".join(parts) if parts else "❌ No relevant information found."

# --------------------------------------------------
# Answer Generation (LLM-based)
# --------------------------------------------------
def generate_answer_llm(query: str, vec, graph, tri):
    """Generate answer using Ollama LLM"""
    
    if not USE_LLM:
        return generate_answer_template(vec, graph, tri), False
    
    try:
        import requests
        
        # Build context
        context_parts = []
        
        if vec:
            context_parts.append("Documents:")
            for r in vec[:3]:
                context_parts.append(f"- {r['content'][:200]}")
        
        if tri:
            context_parts.append("\nFacts:")
            for t in tri[:5]:
                context_parts.append(f"- {t['subject']} {t['predicate']} {t['object']}")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following information, answer the question concisely.

Context:
{context}

Question: {query}

Answer (be brief and factual):"""
        
        # Call Ollama
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 200
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            return answer, True
        else:
            logger.warning(f"LLM returned status {response.status_code}")
            return generate_answer_template(vec, graph, tri), False
    
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return generate_answer_template(vec, graph, tri), False

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/")
def index():
    """Serve main page"""
    try:
        return render_template("chatbot.html")
    except Exception as e:
        logger.error(f"Template error: {e}")
        return jsonify({"error": "Template not found"}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint
    
    ✅ FIXED: Proper error handling, validation, logging
    """
    start_total = time.time()

    try:
        # Validate request
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get("query", "").strip()
        search_type = data.get("search_type", "hybrid")
        
        # ✅ Input validation
        if not query:
            return jsonify({"error": "Empty query"}), 400
        
        if len(query) > 500:
            return jsonify({"error": "Query too long (max 500 chars)"}), 400
        
        logger.info(f"Query: '{query}' | Type: {search_type}")

        # Initialize results
        vector_results, vector_latency = [], 0
        graph_results, graph_latency = [], 0
        triple_results = []

        # Perform searches based on type
        if search_type in ["vector", "hybrid"]:
            vector_results, vector_latency = vector_search(
                query, 
                EMBEDDING_CONFIG["top_k"]
            )

        if search_type in ["graph", "hybrid"]:
            graph_results, graph_latency = graph_search(query)
            triple_results = triple_search(query)

        # Generate answer
        answer_start = time.time()
        answer, llm_used = generate_answer_llm(
            query, 
            vector_results, 
            graph_results, 
            triple_results
        )
        answer_latency = (time.time() - answer_start) * 1000

        total_latency = (time.time() - start_total) * 1000

        # Collect source files
        source_files = sorted(set(
            [v["file"] for v in vector_results] +
            [t["source_file"] for t in triple_results]
        ))

        response = {
            "query": query,
            "answer": answer,
            "llm_used": llm_used,
            "results": {
                "vector": vector_results,
                "graph": graph_results,
                "triples": triple_results
            },
            "metrics": {
                "total_latency_ms": round(total_latency, 2),
                "vector_latency_ms": vector_latency,
                "graph_latency_ms": graph_latency,
                "answer_latency_ms": round(answer_latency, 2),
                "timestamp_utc": datetime.utcnow().isoformat() + "Z"
            },
            "sources": {
                "files": source_files,
                "vector_count": len(vector_results),
                "triple_count": len(triple_results),
                "graph_count": len(graph_results)
            }
        }

        logger.info(f"Response generated in {total_latency:.2f}ms")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route("/api/stats")
def stats():
    """
    Get system statistics
    
    ✅ FIXED: Better error handling
    """
    try:
        # Get Neo4j stats if available
        node_count = 0
        if neo4j_driver:
            try:
                with neo4j_driver.session(database=NEO4J_CONFIG["database"]) as session:
                    result = session.run("MATCH (n) RETURN count(n) AS c")
                    node_count = result.single()["c"]
            except Exception as e:
                logger.warning(f"Failed to get Neo4j stats: {e}")

        llm_status = "connected" if USE_LLM else "disabled"
        
        return jsonify({
            "total_chunks": len(chunks),
            "total_triples": len(triples),
            "total_nodes": node_count,
            "llm_enabled": USE_LLM,
            "llm_status": llm_status,
            "llm_model": OLLAMA_MODEL if USE_LLM else None,
            "embedding_model": EMBEDDING_CONFIG["model_name"],
            "neo4j_connected": neo4j_driver is not None
        })
    
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        return jsonify({
            "error": "Failed to get stats",
            "message": str(e)
        }), 500

@app.route("/api/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "embedding_model": model is not None,
            "faiss_index": faiss_index is not None,
            "neo4j": neo4j_driver is not None,
            "triples": len(triples) > 0,
            "chunks": len(chunks) > 0
        }
    })

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    try:
        # Initialize system
        stats = initialize()
        
        # Show startup info
        logger.info("\n" + "=" * 60)
        logger.info("RAG CHATBOT SERVER")
        logger.info("=" * 60)
        logger.info(f"Chunks loaded: {stats['chunks']}")
        logger.info(f"Triples loaded: {stats['triples']}")
        logger.info(f"Neo4j connected: {stats['neo4j']}")
        logger.info(f"LLM enabled: {USE_LLM}")
        logger.info("=" * 60)
        logger.info("🚀 Server starting at http://localhost:5000")
        logger.info("=" * 60)
        
        # Start server
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=os.getenv("FLASK_DEBUG", "false").lower() == "true"
        )
    
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)