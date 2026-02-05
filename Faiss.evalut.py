import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_PATH = Path(r"C:\Users\ashwa\OneDrive\Desktop\AI_Graph\Internship")

# ✅ CORRECT FILE NAME (matches your pipeline)
TRIPLES_FILE = BASE_PATH / "entity_relation_entity_triples.json"

FAISS_INDEX_FILE = BASE_PATH / "knowledge_graph.index"
TRIPLE_TEXT_FILE = BASE_PATH / "triple_texts.json"
TRIPLE_META_FILE = BASE_PATH / "triples_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

# --------------------------------------------------
# Safety Check
# --------------------------------------------------
if not TRIPLES_FILE.exists():
    raise FileNotFoundError(f"❌ Triples file not found: {TRIPLES_FILE}")

# --------------------------------------------------
# Load Triples
# --------------------------------------------------
print("Loading triples...")
with open(TRIPLES_FILE, "r", encoding="utf-8") as f:
    triples = json.load(f)

print(f"✓ Loaded {len(triples)} triples")

# --------------------------------------------------
# Create Text Representations (head–relation–tail)
# --------------------------------------------------
print("\nCreating triple texts...")
triple_texts = []
valid_triples = []

for t in triples:
    head = t.get("head")
    relation = t.get("relation")
    tail = t.get("tail")

    if not head or not relation or not tail:
        continue  # skip invalid entries

    triple_texts.append(f"{head} {relation} {tail}")
    valid_triples.append(t)

print(f"✓ Created {len(triple_texts)} triple texts")

# Save metadata
with open(TRIPLE_TEXT_FILE, "w", encoding="utf-8") as f:
    json.dump(triple_texts, f, indent=2, ensure_ascii=False)

with open(TRIPLE_META_FILE, "w", encoding="utf-8") as f:
    json.dump(valid_triples, f, indent=2, ensure_ascii=False)

print("✓ Triple texts & metadata saved")

# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------
print("\nLoading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("✓ Model loaded")

# --------------------------------------------------
# Generate Embeddings
# --------------------------------------------------
print("\nGenerating embeddings...")
embeddings = model.encode(triple_texts, show_progress_bar=True)
embeddings = np.asarray(embeddings).astype("float32")

# ✅ Normalize embeddings (IMPORTANT)
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
print(f"✓ Embeddings generated | Dimension: {dimension}")

# --------------------------------------------------
# Build FAISS Index
# --------------------------------------------------
print("\nBuilding FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, str(FAISS_INDEX_FILE))

print(f"✓ Indexed {index.ntotal} vectors")
print(f"✓ FAISS index saved → {FAISS_INDEX_FILE}")

# --------------------------------------------------
# Semantic Search
# --------------------------------------------------
def search_triples(query, k=TOP_K):
    print(f"\n🔍 Query: {query}")

    query_vector = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, k)

    print(f"\nTop {k} Results:")
    print("-" * 60)

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        similarity = 1 / (1 + dist)
        print(f"{rank}. {triple_texts[idx]}")
        print(f"   Similarity: {similarity:.4f}")

    return indices[0], distances[0]

# --------------------------------------------------
# Demo Queries
# --------------------------------------------------
print("\n" + "=" * 60)
print("🚀 FAISS KG VECTOR SEARCH DEMO")
print("=" * 60)

queries = [
    "founder of a company",
    "capital city",
    "technology organization"
]

for q in queries:
    search_triples(q, k=3)

print("\n✅ PIPELINE COMPLETE")