import json
from pathlib import Path

BASE_DIR = Path(r"C:\Users\ashwa\OneDrive\Desktop\AI_Graph\Internship")

INPUT_FILE = BASE_DIR / "entity_relation_entity_triples.json"
OUTPUT_FILE = BASE_DIR / "chunks.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    triples = json.load(f)

chunks = []

for i, t in enumerate(triples):
    text = f"{t['subject']} {t['predicate']} {t['object']}"

    chunks.append({
        "chunk_id": i,
        "content": text,
        "source": t.get("source_file", "kg")
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"✅ Created {len(chunks)} chunks")