import json
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from Config import BASE_DIR
except ImportError:
    print("⚠️ Config.py not found, using current directory")
    BASE_DIR = Path(__file__).parent

# --------------------------------------------------
# Paths
# --------------------------------------------------
RELATIONSHIP_FILE = BASE_DIR / "extracted_relationships_clean.json"
TRIPLE_JSON = BASE_DIR / "entity_relation_entity_triples.json"
TRIPLE_TXT = BASE_DIR / "entity_relation_entity_triples.txt"

# --------------------------------------------------
# STANDARDIZED Triple Schema
# --------------------------------------------------
# ALL triples use: subject, predicate, object
# This ensures compatibility across the entire system
# --------------------------------------------------

def create_triples():
    """
    Create standardized triples from extracted relationships
    
    IMPORTANT: Uses standard schema:
    - subject (entity 1)
    - predicate (relationship)
    - object (entity 2)
    """
    
    print("=" * 60)
    print("TRIPLE CREATION - STANDARDIZED SCHEMA")
    print("=" * 60)
    
    if not RELATIONSHIP_FILE.exists():
        print(f"❌ Relationship file not found: {RELATIONSHIP_FILE}")
        print("\nPlease run Relationship_Ext.py first")
        return
    
    with open(RELATIONSHIP_FILE, "r", encoding="utf-8") as f:
        relationships = json.load(f)
    
    print(f"✅ Loaded {len(relationships)} relationships")

    triples = []
    seen = set()
    
    # Quality filters
    weak_predicates = {"be", "have", "do", "make", "get", "say"}

    for rel in relationships:
        subject = rel.get("subject", "").strip()
        predicate = rel.get("predicate", "").strip().lower()
        obj = rel.get("object", "").strip()

        # Validation
        if not subject or not predicate or not obj:
            continue
        
        if len(subject) < 2 or len(obj) < 2:
            continue

        # Filter weak predicates
        if predicate in weak_predicates:
            continue

        # Deduplication
        key = (subject.lower(), predicate, obj.lower())
        if key in seen:
            continue
        seen.add(key)

        # ✅ STANDARDIZED SCHEMA
        triples.append({
            "subject": subject,           # Entity 1
            "predicate": predicate,       # Relationship
            "object": obj,                # Entity 2
            "subject_type": rel.get("subject_type", "UNKNOWN"),
            "object_type": rel.get("object_type", "UNKNOWN"),
            "sentence": rel.get("sentence", ""),
            "source_file": rel.get("source_file", "unknown"),
            "extraction_method": rel.get("extraction_method", "unknown"),
            "confidence": 1.0
        })

    # --------------------------------------------------
    # Save JSON
    # --------------------------------------------------
    with open(TRIPLE_JSON, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------
    # Save TXT (Human-readable)
    # --------------------------------------------------
    with open(TRIPLE_TXT, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(f"{t['subject']} --[{t['predicate']}]--> {t['object']}\n")

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------
    print(f"\n✅ Created {len(triples)} triples")
    print(f"✅ Saved JSON → {TRIPLE_JSON}")
    print(f"✅ Saved TXT  → {TRIPLE_TXT}")
    
    # Predicate statistics
    from collections import Counter
    predicates = [t['predicate'] for t in triples]
    top_predicates = Counter(predicates).most_common(10)
    
    print("\n📊 Top Predicates:")
    for pred, count in top_predicates:
        print(f"  {pred:20}: {count:4}")
    
    # Type statistics
    subject_types = Counter(t['subject_type'] for t in triples)
    object_types = Counter(t['object_type'] for t in triples)
    
    print("\n📊 Entity Types:")
    print("  Subject types:", dict(subject_types.most_common(5)))
    print("  Object types: ", dict(object_types.most_common(5)))

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    try:
        create_triples()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)