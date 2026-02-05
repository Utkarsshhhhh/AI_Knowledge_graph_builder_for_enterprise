from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Anand@1234"   # 🔴 update if changed
NEO4J_DB = "neo4j"

print("\nConnecting to Neo4j...")

try:
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    # 🔹 Test authentication explicitly
    with driver.session(database=NEO4J_DB) as session:
        session.run("RETURN 1").consume()

    print("✓ Authentication successful")

except AuthError:
    print("❌ Authentication failed")
    print("➡ Check username/password in Neo4j Browser")
    exit(1)

except ServiceUnavailable:
    print("❌ Neo4j service not running")
    print("➡ Start Neo4j Desktop / Server")
    exit(1)

print("\n" + "=" * 60)
print("KNOWLEDGE GRAPH VALIDATION")
print("=" * 60)

with driver.session(database=NEO4J_DB) as session:

    print("\n1. Total Nodes:")
    result = session.run("""
        MATCH (n:Entity)
        RETURN count(n) AS count
    """)
    print(f"   {result.single()['count']}")

    print("\n2. Total Relationships:")
    result = session.run("""
        MATCH ()-[r:RELATES]->()
        RETURN count(r) AS count
    """)
    print(f"   {result.single()['count']}")

    print("\n3. Nodes by Type:")
    result = session.run("""
        MATCH (n:Entity)
        RETURN n.type AS type, count(n) AS count
        ORDER BY count DESC
    """)
    for r in result:
        print(f"   {r['type']}: {r['count']}")

    print("\n4. Top Relationship Predicates:")
    result = session.run("""
        MATCH ()-[r:RELATES]->()
        RETURN r.predicate AS predicate, count(r) AS count
        ORDER BY count DESC LIMIT 5
    """)
    for r in result:
        print(f"   {r['predicate']}: {r['count']}")

    print("\n5. Sample Triples:")
    result = session.run("""
        MATCH (s:Entity)-[r:RELATES]->(o:Entity)
        RETURN s.name AS subject, r.predicate AS relation, o.name AS object
        LIMIT 5
    """)
    for r in result:
        print(f"   {r['subject']} --[{r['relation']}]--> {r['object']}")

    print("\n6. Most Connected Nodes:")
    result = session.run("""
        MATCH (n:Entity)
        RETURN n.name AS name, COUNT { (n)--() } AS connections
        ORDER BY connections DESC LIMIT 5
    """)
    for r in result:
        print(f"   {r['name']}: {r['connections']} connections")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE!")
print("=" * 60)

driver.close()