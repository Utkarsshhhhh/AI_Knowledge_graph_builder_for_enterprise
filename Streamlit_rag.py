"""
Production-Ready Streamlit UI for Your RAG System
Works with your existing Config.py, vector_database.index, and Neo4j setup

Installation:
    pip install streamlit plotly

Usage:
    streamlit run streamlit_rag_ui.py
"""

import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
from datetime import datetime
import plotly.graph_objects as go
from neo4j import GraphDatabase

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import your configuration
try:
    from Config import (
        BASE_DIR, NEO4J_CONFIG, EMBEDDING_CONFIG,
        VECTOR_INDEX_PATH, VECTOR_METADATA_PATH,
        KNOWLEDGE_TRIPLES_PATH
    )
    config_loaded = True
except ImportError:
    st.error("❌ Config.py not found. Please ensure it's in the same directory.")
    st.stop()
    config_loaded = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Knowledge Graph Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    
    /* Confidence badges */
    .confidence-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 600;
        margin-left: 10px;
    }
    
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
    }
    
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Metrics */
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px 0;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #6c757d;
        text-transform: uppercase;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model (cached)"""
    try:
        model = SentenceTransformer(EMBEDDING_CONFIG['model_name'])
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_faiss_index():
    """Load FAISS index and metadata (cached)"""
    try:
        if not VECTOR_INDEX_PATH.exists():
            return None, None, f"FAISS index not found: {VECTOR_INDEX_PATH}"
        
        index = faiss.read_index(str(VECTOR_INDEX_PATH))
        
        with open(VECTOR_METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            chunks = metadata.get('chunks', [])
        
        return index, chunks, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_knowledge_triples():
    """Load knowledge triples (cached)"""
    try:
        if not KNOWLEDGE_TRIPLES_PATH.exists():
            return [], f"Triples file not found: {KNOWLEDGE_TRIPLES_PATH}"
        
        with open(KNOWLEDGE_TRIPLES_PATH, 'r', encoding='utf-8') as f:
            triples = json.load(f)
        
        return triples, None
    except Exception as e:
        return [], str(e)

def get_neo4j_driver():
    """Create Neo4j driver connection"""
    try:
        driver = GraphDatabase.driver(
            NEO4J_CONFIG['uri'],
            auth=(NEO4J_CONFIG['user'], NEO4J_CONFIG['password'])
        )
        # Test connection
        with driver.session(database=NEO4J_CONFIG['database']) as session:
            session.run("RETURN 1").single()
        return driver, None
    except Exception as e:
        return None, str(e)

# ============================================================================
# QUERY PROCESSING FUNCTIONS
# ============================================================================

def search_vectors(query, model, index, chunks, top_k=5):
    """Search FAISS index for similar chunks"""
    try:
        # Encode query
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = index.search(query_embedding, top_k)
        
        # Collect results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(chunks):
                chunk = chunks[idx]
                results.append({
                    'content': chunk.get('content', ''),
                    'file': chunk.get('file', 'Unknown'),
                    'chunk_id': chunk.get('chunk_id', idx),
                    'similarity': float(dist)
                })
        
        return results, None
    except Exception as e:
        return [], str(e)

def search_knowledge_graph(query, neo4j_driver, limit=5):
    """Search Neo4j knowledge graph"""
    try:
        with neo4j_driver.session(database=NEO4J_CONFIG['database']) as session:
            # Simple text search on entity names
            result = session.run("""
                MATCH (s:Entity)-[r:RELATES]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($query)
                   OR toLower(o.name) CONTAINS toLower($query)
                   OR toLower(r.predicate) CONTAINS toLower($query)
                RETURN s.name as subject, r.predicate as relation, o.name as object
                LIMIT $limit
            """, query=query, limit=limit)
            
            triples = []
            for record in result:
                triples.append({
                    'subject': record['subject'],
                    'relation': record['relation'],
                    'object': record['object']
                })
            
            return triples, None
    except Exception as e:
        return [], str(e)

def calculate_confidence(vector_results, graph_results):
    """Calculate confidence score based on results"""
    if not vector_results and not graph_results:
        return 0.0
    
    # Base confidence from vector similarity
    if vector_results:
        avg_similarity = np.mean([r['similarity'] for r in vector_results])
        base_confidence = avg_similarity
    else:
        base_confidence = 0.3
    
    # Boost for graph matches
    graph_boost = min(len(graph_results) * 0.1, 0.3)
    
    confidence = base_confidence + graph_boost
    return min(1.0, max(0.0, confidence))

def generate_answer(query, vector_results, graph_results):
    """Generate answer from retrieved results"""
    if not vector_results and not graph_results:
        return "I couldn't find relevant information to answer your question."
    
    # Start with vector search results
    answer_parts = []
    
    if vector_results:
        top_result = vector_results[0]
        content = top_result['content']
        
        # Truncate if too long
        if len(content) > 300:
            content = content[:300] + "..."
        
        answer_parts.append(f"Based on the documents:\n\n{content}")
    
    # Add knowledge graph results
    if graph_results:
        answer_parts.append("\n\nRelated facts from knowledge graph:")
        for triple in graph_results[:3]:
            answer_parts.append(
                f"- {triple['subject']} {triple['relation']} {triple['object']}"
            )
    
    return "\n".join(answer_parts)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_queries': 0,
        'successful_queries': 0,
        'failed_queries': 0,
        'avg_confidence': 0.0,
        'total_latency': 0.0
    }

if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True

if 'show_graph_results' not in st.session_state:
    st.session_state.show_graph_results = True

# ============================================================================
# LOAD ALL RESOURCES
# ============================================================================

with st.spinner("🔄 Loading knowledge system..."):
    # Load embedding model
    model, model_error = load_embedding_model()
    if model_error:
        st.error(f"❌ Failed to load embedding model: {model_error}")
        st.stop()
    
    # Load FAISS index
    index, chunks, index_error = load_faiss_index()
    if index_error:
        st.error(f"❌ Failed to load FAISS index: {index_error}")
        st.stop()
    
    # Load knowledge triples
    triples, triples_error = load_knowledge_triples()
    if triples_error:
        st.warning(f"⚠️ Could not load knowledge triples: {triples_error}")
        triples = []
    
    # Connect to Neo4j
    neo4j_driver, neo4j_error = get_neo4j_driver()
    if neo4j_error:
        st.warning(f"⚠️ Neo4j not available: {neo4j_error}")
        neo4j_driver = None

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1>🧠 Knowledge Graph Assistant</h1>
    <p>Ask questions about your data • Powered by Vector Search & Knowledge Graphs</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 System Status")
    
    # System metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(chunks):,}</div>
            <div class="metric-label">Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(triples):,}</div>
            <div class="metric-label">Triples</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Session statistics
    st.header("📈 Session Stats")
    st.metric("Total Queries", st.session_state.stats['total_queries'])
    st.metric("Success Rate", 
              f"{(st.session_state.stats['successful_queries'] / max(1, st.session_state.stats['total_queries']) * 100):.1f}%")
    
    if st.session_state.stats['total_queries'] > 0:
        avg_latency = st.session_state.stats['total_latency'] / st.session_state.stats['total_queries']
        st.metric("Avg Response Time", f"{avg_latency:.0f}ms")
        st.metric("Avg Confidence", f"{st.session_state.stats['avg_confidence']:.0%}")
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    st.session_state.show_sources = st.checkbox("Show Sources", value=True)
    st.session_state.show_graph_results = st.checkbox("Show Graph Results", value=True)
    top_k = st.slider("Results per query", 1, 10, 5)
    
    st.divider()
    
    # Actions
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reset Stats"):
        st.session_state.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_confidence': 0.0,
            'total_latency': 0.0
        }
        st.rerun()

# ============================================================================
# CHAT INTERFACE
# ============================================================================

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant":
            if "confidence" in message:
                conf = message["confidence"]
                if conf >= 0.8:
                    badge_html = '<span class="confidence-badge confidence-high">🟢 High Confidence</span>'
                elif conf >= 0.5:
                    badge_html = '<span class="confidence-badge confidence-medium">🟡 Medium Confidence</span>'
                else:
                    badge_html = '<span class="confidence-badge confidence-low">🔴 Low Confidence</span>'
                
                st.markdown(f"{badge_html} ({conf:.1%})", unsafe_allow_html=True)
            
            # Show sources
            if st.session_state.show_sources and "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. {source['file']}** (similarity: {source['similarity']:.3f})")
                        preview = source['content'][:150]
                        if len(source['content']) > 150:
                            preview += "..."
                        st.markdown(f"> {preview}")
            
            # Show graph results
            if st.session_state.show_graph_results and "graph_results" in message and message["graph_results"]:
                with st.expander("🔗 Knowledge Graph Matches"):
                    for triple in message["graph_results"]:
                        st.markdown(f"• **{triple['subject']}** --[{triple['relation']}]--> **{triple['object']}**")

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            import time
            start_time = time.time()
            
            # Search vector database
            vector_results, vector_error = search_vectors(prompt, model, index, chunks, top_k)
            
            # Search knowledge graph (if available)
            graph_results = []
            if neo4j_driver:
                graph_results, graph_error = search_knowledge_graph(prompt, neo4j_driver, limit=5)
            
            # Calculate confidence
            confidence = calculate_confidence(vector_results, graph_results)
            
            # Generate answer
            answer = generate_answer(prompt, vector_results, graph_results)
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000  # ms
            
            # Display answer
            st.markdown(answer)
            
            # Display confidence badge
            if confidence >= 0.8:
                badge_html = '<span class="confidence-badge confidence-high">🟢 High Confidence</span>'
            elif confidence >= 0.5:
                badge_html = '<span class="confidence-badge confidence-medium">🟡 Medium Confidence</span>'
            else:
                badge_html = '<span class="confidence-badge confidence-low">🔴 Low Confidence</span>'
            
            st.markdown(f"{badge_html} ({confidence:.1%})", unsafe_allow_html=True)
            
            # Show low confidence warning
            if confidence < 0.5:
                st.warning("⚠️ This answer has low confidence. Please verify the information.")
            
            # Show sources
            if st.session_state.show_sources and vector_results:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(vector_results, 1):
                        st.markdown(f"**{i}. {source['file']}** (similarity: {source['similarity']:.3f})")
                        preview = source['content'][:150]
                        if len(source['content']) > 150:
                            preview += "..."
                        st.markdown(f"> {preview}")
            
            # Show graph results
            if st.session_state.show_graph_results and graph_results:
                with st.expander("🔗 Knowledge Graph Matches"):
                    for triple in graph_results:
                        st.markdown(f"• **{triple['subject']}** --[{triple['relation']}]--> **{triple['object']}**")
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "confidence": confidence,
        "sources": vector_results[:5],
        "graph_results": graph_results
    })
    
    # Update stats
    st.session_state.stats['total_queries'] += 1
    st.session_state.stats['successful_queries'] += 1 if vector_results or graph_results else 0
    st.session_state.stats['failed_queries'] += 0 if vector_results or graph_results else 1
    st.session_state.stats['total_latency'] += latency
    
    # Update average confidence
    total = st.session_state.stats['total_queries']
    current_avg = st.session_state.stats['avg_confidence']
    st.session_state.stats['avg_confidence'] = (current_avg * (total - 1) + confidence) / total
    
    # Rerun to update sidebar
    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.85em;">
    💡 <strong>Tip:</strong> Ask specific questions for better results •
    Questions about entities, relationships, or facts work best
</div>
""", unsafe_allow_html=True)