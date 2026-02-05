import json
import spacy
import re
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_PATH = Path(r"C:\Users\ashwa\OneDrive\Desktop\AI_Graph\Internship")
INGESTED_DATA_PATH = BASE_PATH / "ingested_data.json"
OUTPUT_PATH = BASE_PATH / "extracted_relationships_clean.json"

# --------------------------------------------------
# Load spaCy Model (Use trf for high accuracy)
# --------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# --------------------------------------------------
# Relationship Patterns
# --------------------------------------------------
RELATIONSHIP_PATTERNS = [
    {
        "pattern": r"(\b\w+(?:\s+\w+)*)\s+(?:founded|created|established)\s+(\b\w+(?:\s+\w+)*)",
        "subject_group": 1,
        "object_group": 2,
        "predicate": "founded"
    },
    {
        "pattern": r"(\b\w+(?:\s+\w+)*)\s+(?:acquired|bought|purchased)\s+(\b\w+(?:\s+\w+)*)",
        "subject_group": 1,
        "object_group": 2,
        "predicate": "acquired"
    },
    {
        "pattern": r"(\b\w+(?:\s+\w+)*)\s+(?:is\s+located|based|headquartered)\s+in\s+(\b\w+(?:\s+\w+)*)",
        "subject_group": 1,
        "object_group": 2,
        "predicate": "located_in"
    }
]

# --------------------------------------------------
# Chunk large text
# --------------------------------------------------
def chunk_text(text, chunk_size=5000):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]

# --------------------------------------------------
# Pattern Extraction
# --------------------------------------------------
def extract_pattern_relationships(text, file_name):

    relationships = []
    seen = set()

    for pattern_info in RELATIONSHIP_PATTERNS:

        pattern = re.compile(pattern_info["pattern"], re.IGNORECASE)

        for match in pattern.finditer(text):

            subject = match.group(pattern_info["subject_group"]).strip()
            obj = match.group(pattern_info["object_group"]).strip()
            predicate = pattern_info["predicate"]

            key = (subject.lower(), predicate, obj.lower())

            if key in seen:
                continue
            seen.add(key)

            relationships.append({
                "subject": subject,
                "subject_type": "UNKNOWN",
                "predicate": predicate,
                "object": obj,
                "object_type": "UNKNOWN",
                "sentence": match.group(0),
                "confidence": 0.9,
                "source_file": file_name,
                "extraction_method": "pattern"
            })

    return relationships

# --------------------------------------------------
# High Accuracy Dependency Extraction
# --------------------------------------------------
def extract_high_accuracy_relationships(text, file_name):

    relationships = []
    seen = set()

    VALID_PREDICATES = {
        "be","have","own","acquire","buy","found",
        "create","build","lead","head","run",
        "manage","develop","produce","establish",
        "locate","base"
    }

    predicate_mapping = {
        "be":"is",
        "have":"has",
        "acquire":"acquired",
        "buy":"bought",
        "found":"founded",
        "create":"created",
        "build":"built",
        "lead":"leads",
        "head":"heads",
        "run":"runs",
        "manage":"manages",
        "develop":"develops",
        "produce":"produces",
        "establish":"established",
        "locate":"located_in",
        "base":"based_in"
    }

    for doc in nlp.pipe(chunk_text(text), batch_size=10):

        for sent in doc.sents:

            for token in sent:

                if token.dep_ != "ROOT":
                    continue

                predicate = token.lemma_.lower()

                if predicate not in VALID_PREDICATES:
                    continue

                subject = None
                obj = None

                for child in token.lefts:
                    if child.dep_ in ("nsubj","nsubjpass"):
                        subject = child

                for child in token.rights:
                    if child.dep_ in ("dobj","pobj","attr","oprd"):
                        obj = child

                if not subject or not obj:
                    continue

                subject_ent = None
                object_ent = None

                for ent in sent.ents:
                    if subject.text in ent.text:
                        subject_ent = ent
                    if obj.text in ent.text:
                        object_ent = ent

                if not subject_ent or not object_ent:
                    continue

                predicate = predicate_mapping.get(predicate,predicate)

                key = (
                    subject_ent.text.lower(),
                    predicate,
                    object_ent.text.lower()
                )

                if key in seen:
                    continue
                seen.add(key)

                confidence = 0.7
                if subject_ent.label_ in ["ORG","PERSON"]:
                    confidence += 0.1
                if object_ent.label_ in ["ORG","PERSON","GPE"]:
                    confidence += 0.1

                relationships.append({
                    "subject": subject_ent.text,
                    "subject_type": subject_ent.label_,
                    "predicate": predicate,
                    "object": object_ent.text,
                    "object_type": object_ent.label_,
                    "sentence": sent.text.strip(),
                    "confidence": round(confidence,2),
                    "source_file": file_name,
                    "extraction_method": "dependency"
                })

    return relationships

# --------------------------------------------------
# Main Extraction Pipeline
# --------------------------------------------------
def extract_relationships():

    with open(INGESTED_DATA_PATH,"r",encoding="utf-8") as f:
        data = json.load(f)

    all_relationships = []
    stats = defaultdict(int)

    for item in data.get("unstructured",[]):

        file_name = item.get("file","unknown")
        text = item.get("data","")

        if not isinstance(text,str) or len(text)<50:
            continue

        print(f"Processing: {file_name}")

        pattern_rels = extract_pattern_relationships(text,file_name)
        dep_rels = extract_high_accuracy_relationships(text,file_name)

        combined = pattern_rels + dep_rels
        all_relationships.extend(combined)

        print("  Pattern:",len(pattern_rels))
        print("  Dependency:",len(dep_rels))

        for rel in combined:
            stats[rel["predicate"]] += 1

    unique = []
    seen=set()

    for rel in all_relationships:
        key=(rel["subject"].lower(),rel["predicate"],rel["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(rel)

    with open(OUTPUT_PATH,"w",encoding="utf-8") as f:
        json.dump(unique,f,indent=2,ensure_ascii=False)

    print("\n✅ Extracted:",len(unique))
    print("\nTop predicates:")
    for k,v in sorted(stats.items(),key=lambda x:x[1],reverse=True)[:10]:
        print(k,":",v)

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__=="__main__":
    extract_relationships()