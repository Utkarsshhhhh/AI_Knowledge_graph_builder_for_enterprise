import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import sys

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = r"C:\Users\ashwa\OneDrive\Desktop\AI_Graph\Internship"

VECTOR_INDEX = os.path.join(BASE_DIR, "vector_database.index")
VECTOR_METADATA = os.path.join(BASE_DIR, "vector_metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------
# Load Metadata (supports dict + list)
# --------------------------------------------------
def load_metadata():
    try:
        with open(VECTOR_METADATA, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Support both metadata structures
        if isinstance(metadata, dict):
            chunks = metadata.get("chunks", [])
        elif isinstance(metadata, list):
            chunks = metadata
        else:
            raise ValueError("Invalid metadata structure")

        if not chunks:
            raise ValueError("No chunks found in metadata")

        print(f"✓ Loaded {len(chunks)} chunks")
        return chunks

    except Exception as e:
        print(f"\n❌ Error loading metadata: {e}")
        sys.exit(1)


# --------------------------------------------------
# Search Function
# --------------------------------------------------
def perform_search(index, model, chunks, query, top_k=5, show_full=False):

    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")

    # IMPORTANT: normalize for cosine similarity
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, top_k)

    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60 + "\n")

    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):

        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        content = chunk.get("content", "")

        if not show_full and len(content) > 200:
            content = content[:200] + "..."

        file_name = chunk.get("file", "Unknown")
        chunk_id = chunk.get("chunk_id", idx)

        print(f"{rank}. [{file_name}] Chunk #{chunk_id} | Similarity: {score:.4f}")
        print(f"   {content}\n")


# --------------------------------------------------
# Main Function
# --------------------------------------------------
def main():

    print("="*60)
    print("📄 DOCUMENT SEARCH SYSTEM")
    print("="*60)
    print("\nInitializing...")

    # Check files
    if not os.path.exists(VECTOR_INDEX):
        print(f"\n❌ FAISS index missing: {VECTOR_INDEX}")
        sys.exit(1)

    if not os.path.exists(VECTOR_METADATA):
        print(f"\n❌ Metadata missing: {VECTOR_METADATA}")
        sys.exit(1)

    # Load model
    print("\nLoading embedding model...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("✓ Model loaded")
    except Exception as e:
        print(f"\n❌ Model load error: {e}")
        sys.exit(1)

    # Load FAISS
    print("\nLoading FAISS index...")
    try:
        index = faiss.read_index(VECTOR_INDEX)
        print(f"✓ Loaded index ({index.ntotal} vectors)")
    except Exception as e:
        print(f"\n❌ FAISS load error: {e}")
        sys.exit(1)

    # Load metadata
    chunks = load_metadata()

    if index.ntotal != len(chunks):
        print(f"\n⚠️ Warning: index vectors ({index.ntotal}) != chunks ({len(chunks)})")

    # Interactive loop
    print("\n" + "="*60)
    print("🔍 SEARCH READY")
    print("="*60)
    print("Type query | full:query for full text | exit to quit\n")

    while True:

        try:
            query = input("🔍 Search: ").strip()

            if query.lower() in {"exit", "quit", "q"}:
                break

            if not query:
                continue

            show_full = query.lower().startswith("full:")
            clean_query = query[5:].strip() if show_full else query

            perform_search(index, model, chunks, clean_query, show_full=show_full)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Runtime error: {e}")

    print("\n👋 Goodbye!")


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    main()