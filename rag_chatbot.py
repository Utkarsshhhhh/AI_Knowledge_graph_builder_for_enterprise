"""
Enhanced RAG Backend with Ollama LLM Integration
- Query preprocessing and expansion
- Entity recognition in queries
- Semantic reranking
- Natural language answer generation using Ollama
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
from neo4j import GraphDatabase
import os
import re
import requests

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = r"C:\Users\ashwa\OneDrive\Desktop\AI_Graph\Internship"

VECTOR_INDEX = os.path.join(BASE_DIR, "vector_database.index")
VECTOR_METADATA = os.path.join(BASE_DIR, "vector_metadata.json")
KNOWLEDGE_TRIPLES = os.path.join(
    BASE_DIR, "entity_relation_entity_triples.json"
)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Anand@1234"

# Ollama Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # or "mistral", "llama2", "phi", etc.
USE_OLLAMA = True  # Set to False to disable LLM

# --------------------------------------------------
# Query expansion patterns
# --------------------------------------------------
QUERY_EXPANSIONS = {
    "ceo": ["chief executive officer", "ceo", "chief executive", "head", "leader"],
    "founder": ["founder", "co-founder", "established", "created", "started"],
    "capital": ["capital", "capital city", "main city"],
    "location": ["located", "based in", "headquarters", "hq"],
    "product": ["product", "makes", "produces", "manufactures"],
    "owner": ["owner", "owns", "acquired", "purchased"]
}

# Predicate synonyms for better matching
PREDICATE_SYNONYMS = {
    "ceo": ["is_ceo_of", "has_ceo", "heads", "leads", "runs", "manages"],
    "founder": ["founded", "co-founded", "established", "created", "started"],
    "capital": ["has_capital", "capital_of"],
    "located": ["located_in", "based_in", "headquarters_in"]
}

# Common stop words
STOP_WORDS = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 
              'were', 'the', 'a', 'an', 'of', 'in', 'to', 'for', 'on', 'at', 
              'from', 'by', 'about', 'as', 'with', 'into', 'tell', 'me'}

# --------------------------------------------------
# Globals
# --------------------------------------------------
model = None
faiss_index = None
chunks = []
triples = []
neo4j_driver = None

# --------------------------------------------------
# Initialization
# --------------------------------------------------
def initialize():
    global model, faiss_index, chunks, triples, neo4j_driver

    print("Initializing Enhanced RAG Backend with Ollama...")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load FAISS index
    faiss_index = faiss.read_index(VECTOR_INDEX)

    # Load metadata
    with open(VECTOR_METADATA, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    # Load triples
    with open(KNOWLEDGE_TRIPLES, "r", encoding="utf-8") as f:
        triples = json.load(f)

    # Connect to Neo4j
    neo4j_driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    print(f"✅ Loaded {len(chunks)} chunks")
    print(f"✅ Loaded {len(triples)} triples")
    
    # Test Ollama connection
    if USE_OLLAMA:
        try:
            test_response = requests.post(
                OLLAMA_API_URL,
                json={"model": OLLAMA_MODEL, "prompt": "test", "stream": False},
                timeout=5
            )
            if test_response.status_code == 200:
                print(f"✅ Ollama connected (model: {OLLAMA_MODEL})")
            else:
                print(f"⚠️  Ollama API returned status {test_response.status_code}")
                print("   Falling back to template-based answers")
        except Exception as e:
            print(f"⚠️  Ollama not available: {e}")
            print("   Make sure Ollama is running: ollama serve")
            print("   Falling back to template-based answers")
    
    print("✅ Initialization complete")

# --------------------------------------------------
# Query Processing (No spaCy - Python 3.14 compatible)
# --------------------------------------------------
def process_query(query):
    """Process and enhance query for better matching"""
    query_lower = query.lower().strip()
    
    result = {
        "original": query,
        "normalized": query_lower,
        "keywords": [],
        "intent": None,
        "expanded_terms": []
    }
    
    # Simple tokenization
    words = re.findall(r'\b\w+\b', query_lower)
    result["keywords"] = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    # Detect query intent
    if any(term in query_lower for term in ["ceo", "chief executive"]):
        result["intent"] = "ceo"
        result["expanded_terms"].extend(QUERY_EXPANSIONS["ceo"])
    elif any(term in query_lower for term in ["founder", "founded", "co-founder"]):
        result["intent"] = "founder"
        result["expanded_terms"].extend(QUERY_EXPANSIONS["founder"])
    elif any(term in query_lower for term in ["capital", "capital city"]):
        result["intent"] = "capital"
        result["expanded_terms"].extend(QUERY_EXPANSIONS["capital"])
    elif any(term in query_lower for term in ["president", "chairman"]):
        result["intent"] = "president"
    elif any(term in query_lower for term in ["located", "location", "where", "headquarters"]):
        result["intent"] = "location"
        result["expanded_terms"].extend(QUERY_EXPANSIONS["location"])
    
    return result

# --------------------------------------------------
# Enhanced Vector Search
# --------------------------------------------------
def vector_search(query, processed_query, top_k=10):
    """Enhanced vector search with query expansion"""
    start = time.time()
    
    # Search with original query
    q_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)
    distances, indices = faiss_index.search(q_vec, top_k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        c = chunks[idx]
        base_score = 1 / (1 + dist)
        
        # Boost score if chunk contains keywords
        boost = 1.0
        content_lower = c["content"].lower()
        
        for keyword in processed_query["keywords"]:
            if keyword in content_lower:
                boost *= 1.1
        
        final_score = min(base_score * boost, 1.0)
        
        results.append({
            "file": c["file"],
            "chunk_id": c["chunk_id"],
            "content": c["content"],
            "score": round(final_score, 3)
        })
    
    # Sort by final score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    latency = (time.time() - start) * 1000
    return results[:top_k], round(latency, 2)

# --------------------------------------------------
# Enhanced Graph Search
# --------------------------------------------------
def graph_search(query, processed_query, limit=10):
    """Enhanced graph search with entity focus"""
    start = time.time()
    
    search_terms = processed_query["keywords"]
    
    with neo4j_driver.session(database="neo4j") as session:
        cypher = """
        MATCH (n:Entity)
        WHERE ANY(term IN $terms WHERE toLower(n.name) CONTAINS toLower(term))
        OPTIONAL MATCH (n)-[r:RELATES]-(m:Entity)
        RETURN n.name AS entity,
               n.type AS type,
               collect(DISTINCT {
                   name: m.name,
                   relation: r.predicate,
                   type: m.type
               }) AS connections
        ORDER BY size(connections) DESC
        LIMIT $limit
        """
        
        records = session.run(cypher, terms=search_terms, limit=limit)
        
        results = []
        for r in records:
            results.append({
                "entity": r["entity"],
                "type": r["type"],
                "connections": [c for c in r["connections"] if c["name"]]
            })
    
    latency = (time.time() - start) * 1000
    return results, round(latency, 2)

# --------------------------------------------------
# Enhanced Triple Search with Reranking
# --------------------------------------------------
def triple_search(query, processed_query, limit=20):
    """Enhanced triple search with relevance scoring"""
    
    # Get relevant predicates based on intent
    relevant_predicates = []
    if processed_query["intent"] and processed_query["intent"] in PREDICATE_SYNONYMS:
        relevant_predicates = PREDICATE_SYNONYMS[processed_query["intent"]]
    
    keywords = set(processed_query["keywords"])
    scored_triples = []
    
    for t in triples:
        score = 0
        head_lower = t["head"].lower()
        tail_lower = t["tail"].lower()
        relation_lower = t["relation"].lower()
        
        # Score based on keyword match
        for keyword in keywords:
            if keyword in head_lower:
                score += 5
            if keyword in tail_lower:
                score += 5
            if keyword in relation_lower:
                score += 3
        
        # Boost if predicate matches intent
        if relevant_predicates:
            if relation_lower in relevant_predicates:
                score += 10
            # Also check if any relevant predicate is substring
            for pred in relevant_predicates:
                if pred in relation_lower or relation_lower in pred:
                    score += 5
                    break
        
        # Boost for exact keyword matches
        for keyword in keywords:
            if keyword == head_lower or keyword == tail_lower:
                score += 15
        
        if score > 0:
            scored_triples.append({
                "subject": t["head"],
                "predicate": t["relation"],
                "object": t["tail"],
                "source_file": t.get("source_file", "knowledge_graph"),
                "score": score
            })
    
    # Sort by score
    scored_triples.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_triples[:limit]

# --------------------------------------------------
# Ollama LLM Answer Generation
# --------------------------------------------------
def generate_llm_answer(query, context_data):
    """Generate natural language answer using Ollama"""
    
    # Build context from retrieved data
    context_parts = []
    
    # Add triples as facts
    if context_data["triples"]:
        context_parts.append("Knowledge Graph Facts:")
        for i, t in enumerate(context_data["triples"][:5], 1):
            context_parts.append(f"{i}. {t['subject']} {t['predicate']} {t['object']}")
    
    # Add document chunks
    if context_data["documents"]:
        context_parts.append("\nRelevant Documents:")
        for i, doc in enumerate(context_data["documents"][:3], 1):
            snippet = doc["content"][:300]
            context_parts.append(f"{i}. {snippet}")
    
    # Add graph entities
    if context_data["entities"]:
        context_parts.append("\nRelated Entities:")
        for i, ent in enumerate(context_data["entities"][:3], 1):
            conns = ", ".join([c["name"] for c in ent["connections"][:3]])
            context_parts.append(f"{i}. {ent['entity']} ({ent['type']}): connected to {conns}")
    
    context = "\n".join(context_parts)
    
    # Build prompt
    prompt = f"""You are a helpful AI assistant answering questions based on a knowledge base.

Context from Knowledge Base:
{context}

User Question: {query}

Instructions:
1. Answer the question directly and concisely based ONLY on the provided context
2. If the context contains a clear answer, state it confidently
3. If the context doesn't fully answer the question, say what you know and acknowledge what's missing
4. Do not make up information not present in the context
5. Keep your answer under 100 words

Answer:"""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused answers
                    "top_p": 0.9,
                    "num_predict": 150   # Limit response length
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            print(f"Ollama API error: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("Ollama request timed out")
        return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

# --------------------------------------------------
# Template-based Answer Generation (Fallback)
# --------------------------------------------------
def generate_template_answer(query, processed_query, vec, graph, tri):
    """Generate answer using templates (no LLM)"""
    
    # For specific intents, provide direct answer
    if processed_query["intent"] in ["ceo", "founder", "president"]:
        if tri:
            best = tri[0]
            
            if processed_query["intent"] == "ceo":
                return f"**{best['object']}** is the CEO of **{best['subject']}**."
            elif processed_query["intent"] == "founder":
                if "founded_by" in best['predicate']:
                    return f"**{best['subject']}** was founded by **{best['object']}**."
                else:
                    return f"**{best['subject']}** founded **{best['object']}**."
            elif processed_query["intent"] == "president":
                return f"**{best['object']}** is the President of **{best['subject']}**."
    
    # Generate comprehensive answer
    parts = []
    
    if tri:
        parts.append("📌 **Key Facts:**")
        for t in tri[:3]:
            parts.append(f"• {t['subject']} → {t['predicate']} → {t['object']}")
    
    if graph:
        parts.append("\n🔗 **Related Entities:**")
        for g in graph[:3]:
            if g["connections"]:
                conn_str = ", ".join([f"{c['name']}" for c in g["connections"][:3]])
                parts.append(f"• {g['entity']} ({g['type']}): {conn_str}")
    
    if vec:
        parts.append("\n📄 **Supporting Documents:**")
        for r in vec[:2]:
            snippet = r['content'][:150] + "..." if len(r['content']) > 150 else r['content']
            parts.append(f"• {snippet}")
    
    if parts:
        return "\n".join(parts)
    
    return "❌ No relevant information found for your query."

# --------------------------------------------------
# Main Answer Generation
# --------------------------------------------------
def generate_answer(query, processed_query, vec, graph, tri):
    """Generate answer using Ollama or fallback to templates"""
    
    if USE_OLLAMA and (tri or vec):
        # Prepare context for LLM
        context_data = {
            "triples": tri[:5],
            "documents": vec[:3],
            "entities": graph[:3]
        }
        
        llm_answer = generate_llm_answer(query, context_data)
        
        if llm_answer:
            return llm_answer
    
    # Fallback to template-based answer
    return generate_template_answer(query, processed_query, vec, graph, tri)

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("chatbot.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    start_total = time.time()

    data = request.json
    query = data.get("query", "").strip()
    search_type = data.get("search_type", "hybrid")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Process query
    processed_query = process_query(query)
    
    vector_results, vector_latency = [], 0
    graph_results, graph_latency = [], 0
    triple_results = []

    if search_type in ["vector", "hybrid"]:
        vector_results, vector_latency = vector_search(query, processed_query)

    if search_type in ["graph", "hybrid"]:
        graph_results, graph_latency = graph_search(query, processed_query)
        triple_results = triple_search(query, processed_query)

    # Generate answer
    answer_start = time.time()
    answer = generate_answer(query, processed_query, vector_results, graph_results, triple_results)
    answer_latency = (time.time() - answer_start) * 1000

    total_latency = (time.time() - start_total) * 1000

    source_files = sorted(set(
        [v["file"] for v in vector_results] +
        [t["source_file"] for t in triple_results]
    ))

    return jsonify({
        "query": query,
        "processed_query": processed_query,
        "answer": answer,
        "llm_used": USE_OLLAMA and (len(triple_results) > 0 or len(vector_results) > 0),
        "results": {
            "vector": vector_results[:5],
            "graph": graph_results,
            "triples": triple_results[:10]
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
            "triple_count": len(triple_results)
        }
    })

@app.route("/api/stats")
def stats():
    with neo4j_driver.session(database="neo4j") as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]

    # Check Ollama status
    ollama_status = "disabled"
    if USE_OLLAMA:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                ollama_status = f"connected ({OLLAMA_MODEL})"
            else:
                ollama_status = "error"
        except:
            ollama_status = "not running"

    return jsonify({
        "total_chunks": len(chunks),
        "total_triples": len(triples),
        "total_nodes": nodes,
        "llm_enabled": USE_OLLAMA,
        "llm_status": ollama_status,
        "llm_model": OLLAMA_MODEL
    })

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    initialize()
    print("🚀 Enhanced RAG server with Ollama running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)