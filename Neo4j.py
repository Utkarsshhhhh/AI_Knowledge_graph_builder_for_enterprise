"""
Neo4j Knowledge Graph Module
FULLY UPDATED - Neo4j 5.x Compatible
"""

from neo4j import GraphDatabase
import json
from pathlib import Path
import logging

# --------------------------------------------------
# Config
# --------------------------------------------------
try:
    from Config import BASE_DIR, NEO4J_CONFIG, KNOWLEDGE_TRIPLES_PATH
except ImportError:
    print("⚠️ Config.py not found, using defaults")
    BASE_DIR = Path.cwd()
    KNOWLEDGE_TRIPLES_PATH = BASE_DIR / "entity_relation_entity_triples.json"
    NEO4J_CONFIG = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "Anand@1234",
        "database": "neo4j"
    }

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Knowledge Graph Class
# --------------------------------------------------
class KnowledgeGraph:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, session, query, description=""):
        if description:
            logger.info("\n" + "="*60)
            logger.info(description)
            logger.info("="*60)
        return list(session.run(query))

# --------------------------------------------------
# Graph Construction
# --------------------------------------------------
def construct_graph(kg):

    logger.info("\n🗃️ CONSTRUCTION PHASE")

    try:
        with open(KNOWLEDGE_TRIPLES_PATH, "r", encoding="utf-8") as f:
            triples = json.load(f)
        logger.info(f"✅ Loaded {len(triples)} triples")
    except FileNotFoundError:
        logger.error("❌ Triple file not found")
        return

    with kg.driver.session(database=NEO4J_CONFIG["database"]) as session:

        if input("\nClear existing data? (y/N): ").lower() == "y":
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("✅ Database cleared")

        # Constraints
        session.run("""
            CREATE CONSTRAINT entity_name IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """)

        session.run("""
            CREATE INDEX entity_type_idx IF NOT EXISTS
            FOR (e:Entity) ON (e.type)
        """)

        logger.info("✅ Constraints & indexes ready")

    # Import triples
    batch_size = 500

    query = """
    UNWIND $batch AS row
    MERGE (s:Entity {name: row.subject})
    SET s.type = row.subject_type

    MERGE (o:Entity {name: row.object})
    SET o.type = row.object_type

    MERGE (s)-[:RELATES {
        predicate: row.predicate,
        source: row.source_file
    }]->(o)
    """

    logger.info(f"\n📦 Importing {len(triples)} triples...")

    for i in range(0, len(triples), batch_size):

        batch = triples[i:i+batch_size]

        with kg.driver.session(database=NEO4J_CONFIG["database"]) as session:
            session.run(query, batch=batch)

        logger.info(f"  ✅ Imported {min(i+batch_size,len(triples))}/{len(triples)}")

    logger.info("\n✅ Graph construction complete")

# --------------------------------------------------
# Graph Analysis (Neo4j 5 Compatible)
# --------------------------------------------------
def analyze_graph(kg):

    logger.info("\n🔍 ANALYSIS PHASE")

    with kg.driver.session(database=NEO4J_CONFIG["database"]) as session:

        # Graph overview
        nodes = kg.run_query(
            session,
            "MATCH (n:Entity) RETURN count(n) AS count",
            "1. GRAPH OVERVIEW"
        )

        rels = kg.run_query(
            session,
            "MATCH ()-[r:RELATES]->() RETURN count(r) AS count",
            ""
        )

        logger.info(f"Nodes        : {nodes[0]['count']:,}")
        logger.info(f"Relationships: {rels[0]['count']:,}")

        # --------------------------------------------------
        # Top hub entities (FIXED FOR NEO4J 5)
        # --------------------------------------------------
        hubs = kg.run_query(session, """
            MATCH (n:Entity)
            OPTIONAL MATCH (n)--(m)
            RETURN n.name AS name,
                   n.type AS type,
                   COUNT(m) AS degree
            ORDER BY degree DESC
            LIMIT 10
        """, "2. TOP HUB ENTITIES")

        for i,h in enumerate(hubs,1):
            logger.info(f"{i}. {h['name']:<30} ({h['type']}) degree={h['degree']}")

        # --------------------------------------------------
        # Relationship Types
        # --------------------------------------------------
        rel_types = kg.run_query(session, """
            MATCH ()-[r:RELATES]->()
            RETURN r.predicate AS relation,
                   count(*) AS count
            ORDER BY count DESC
        """, "3. RELATIONSHIP TYPES")

        for r in rel_types:
            logger.info(f"  {r['relation']:<15}: {r['count']}")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    logger.info("="*70)
    logger.info("🚀 NEO4J KNOWLEDGE GRAPH - LOAD & ANALYZE")
    logger.info("="*70)

    kg = KnowledgeGraph(
        NEO4J_CONFIG["uri"],
        NEO4J_CONFIG["user"],
        NEO4J_CONFIG["password"]
    )

    try:

        logger.info("🔌 Testing Neo4j connection...")
        with kg.driver.session(database=NEO4J_CONFIG["database"]) as session:
            session.run("RETURN 1")

        logger.info("✅ Connected to Neo4j")

        if input("\n🗃️ Build graph? (y/N): ").lower() == "y":
            construct_graph(kg)

        if input("\n🔍 Run analysis? (y/N): ").lower() == "y":
            analyze_graph(kg)

        logger.info("\n🎉 DONE")

    except Exception as e:
        logger.error(f"❌ Error: {e}")

    finally:
        kg.close()

# --------------------------------------------------
if __name__ == "__main__":
    main()