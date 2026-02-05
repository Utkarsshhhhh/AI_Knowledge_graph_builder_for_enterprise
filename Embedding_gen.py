"""
UPDATED EMBEDDING GENERATION PIPELINE
-------------------------------------
✔ Cosine similarity FAISS (CRITICAL FIX)
✔ Online + Offline model loading
✔ Memory efficient batching
✔ Stable production pipeline
"""

import json
import numpy as np
import faiss
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = Path(r"C:\Users\ashwa\OneDrive\Desktop\AI_Graph\Internship")

DATA_FOLDER = BASE_DIR / "chunks.json"
VECTOR_INDEX_PATH = BASE_DIR / "vector.index"
VECTOR_METADATA_PATH = BASE_DIR / "vector_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL_PATH = BASE_DIR / "local_models" / "all-MiniLM-L6-v2"

BATCH_SIZE = 32

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
def load_embedding_model():
    """Load embedding model (online or offline fallback)"""
    try:
        logging.info("Loading model from HuggingFace...")
        model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(BASE_DIR / "hf_cache")
        )
        return model

    except Exception as e:
        logging.warning("Online model failed. Trying local model...")
        logging.warning(str(e))

        if LOCAL_MODEL_PATH.exists():
            model = SentenceTransformer(str(LOCAL_MODEL_PATH))
            return model
        else:
            raise RuntimeError(
                "❌ Model not found online or locally. "
                "Download from HuggingFace manually."
            )

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
def load_chunks():
    if not DATA_FOLDER.exists():
        raise FileNotFoundError(f"Chunks file not found: {DATA_FOLDER}")

    with open(DATA_FOLDER, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    metadata = []

    for item in data:
        texts.append(item["content"])
        metadata.append(item)

    logging.info(f"Loaded {len(texts)} chunks")
    return texts, metadata

# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------
def create_embeddings(model, texts):
    logging.info("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.astype("float32")

# --------------------------------------------------
# BUILD FAISS INDEX
# --------------------------------------------------
def build_faiss_index(embeddings):
    logging.info("Building FAISS index...")

    # IMPORTANT FIX:
    # Using cosine similarity -> normalized vectors + Inner Product
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    logging.info(f"FAISS index built with {index.ntotal} vectors")
    return index

# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------
def save_outputs(index, metadata):
    logging.info("Saving FAISS index...")
    faiss.write_index(index, str(VECTOR_INDEX_PATH))

    logging.info("Saving metadata...")
    with open(VECTOR_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logging.info("✅ Files saved successfully")

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def main():
    logging.info("🚀 Starting embedding pipeline")

    model = load_embedding_model()

    texts, metadata = load_chunks()

    embeddings = create_embeddings(model, texts)

    index = build_faiss_index(embeddings)

    save_outputs(index, metadata)

    logging.info("✅ Pipeline completed successfully")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()