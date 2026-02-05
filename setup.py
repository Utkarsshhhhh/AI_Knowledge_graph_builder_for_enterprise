"""
Automated Setup Script for RAG Chatbot
Validates environment and helps with configuration
"""

import os
import sys

# Configuration - Update these paths to match your setup
BASE_DIR = r"C:\Users\ashwa\OneDrive\Desktop\AI_Graph\Internship"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ashwan@12"
NEO4J_DATABASE = "neo4j"
SERVER_PORT = 5000

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def check_python_version():
    """Check if Python version is adequate"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} is too old")
        print("  Please install Python 3.8 or higher")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking required packages...")
    
    required_packages = {
        'flask': 'Flask',
        'flask_cors': 'flask-cors',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'neo4j': 'neo4j',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'spacy': 'spacy'
    }
    
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed")
        return True

def check_spacy_model():
    """Check if spaCy model is downloaded"""
    print("\nChecking spaCy model...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy model 'en_core_web_sm' is available")
        return True
    except:
        print("✗ spaCy model 'en_core_web_sm' not found")
        print("\nTo download the model, run:")
        print("  python -m spacy download en_core_web_sm")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("\nChecking data files...")
    
    VECTOR_INDEX_PATH = os.path.join(BASE_DIR, "vector_database.index")
    VECTOR_METADATA_PATH = os.path.join(BASE_DIR, "vector_metadata.json")
    KNOWLEDGE_TRIPLES_PATH = os.path.join(BASE_DIR, "knowledge_triples.json")
    
    print(f"Base directory: {BASE_DIR}")
    
    if not os.path.exists(BASE_DIR):
        print(f"✗ Base directory does not exist: {BASE_DIR}")
        print("\n  Please update BASE_DIR at the top of this script")
        return False
    
    files_to_check = {
        'Vector Index': VECTOR_INDEX_PATH,
        'Vector Metadata': VECTOR_METADATA_PATH,
        'Knowledge Triples': KNOWLEDGE_TRIPLES_PATH
    }
    
    all_exist = True
    for name, path in files_to_check.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✓ {name}: {os.path.basename(path)} ({size:.2f} MB)")
        else:
            print(f"✗ {name}: {os.path.basename(path)} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some data files are missing")
        print("  Please run the data pipeline scripts first to generate these files")
        return False
    else:
        print("\n✓ All required data files exist")
        return True

def check_neo4j_connection():
    """Check if Neo4j is running and accessible"""
    print("\nChecking Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("RETURN 1 as test")
            result.single()
            
            # Get node count
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()['count']
            
            # Get relationship count
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()['count']
        
        driver.close()
        
        print(f"✓ Connected to Neo4j at {NEO4J_URI}")
        print(f"  Database: {NEO4J_DATABASE}")
        print(f"  Nodes: {node_count:,}")
        print(f"  Relationships: {rel_count:,}")
        
        if node_count == 0:
            print("\n⚠️  Warning: Knowledge graph is empty")
            print("  Please load your data into Neo4j first")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j")
        print(f"  Error: {str(e)}")
        print(f"\n  Please ensure:")
        print(f"  1. Neo4j is running")
        print(f"  2. Database '{NEO4J_DATABASE}' exists")
        print(f"  3. Credentials are correct (update at top of this script)")
        print(f"  4. Port is accessible")
        return False

def check_port_available():
    """Check if the server port is available"""
    print("\nChecking server port...")
    
    import socket
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', SERVER_PORT))
        sock.close()
        
        if result == 0:
            print(f"✗ Port {SERVER_PORT} is already in use")
            print(f"\n  Options:")
            print(f"  1. Stop the service using port {SERVER_PORT}")
            print(f"  2. Change SERVER_PORT at top of this script")
            return False
        else:
            print(f"✓ Port {SERVER_PORT} is available")
            return True
    except Exception as e:
        print(f"⚠️  Could not check port {SERVER_PORT}: {e}")
        return True

def suggest_next_steps(checks_passed):
    """Suggest what to do next based on check results"""
    print_header("NEXT STEPS")
    
    if all(checks_passed.values()):
        print("🎉 All checks passed! You're ready to start the chatbot.\n")
        print("To start the basic version:")
        print("  python rag_chatbot_app.py")
        print("\nTo start the advanced version (with LLM support):")
        print("  python rag_chatbot_advanced.py")
        print("\nThen open your browser to:")
        print("  http://localhost:5000")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.\n")
        
        if not checks_passed['dependencies']:
            print("1. Install dependencies:")
            print("   pip install -r requirements.txt")
        
        if not checks_passed['spacy']:
            print("\n2. Download spaCy model:")
            print("   python -m spacy download en_core_web_sm")
        
        if not checks_passed['data_files']:
            print("\n3. Generate data files:")
            print("   Run your data pipeline scripts first")
        
        if not checks_passed['neo4j']:
            print("\n4. Start Neo4j and load data:")
            print("   - Start Neo4j Desktop/Server")
            print("   - Run: python neo4j_load.py (or your graph construction script)")
        
        print("\nAfter fixing issues, run this script again:")
        print("  python setup_check.py")

def main():
    """Main setup check function"""
    print_header("RAG CHATBOT SETUP VALIDATION")
    
    checks_passed = {
        'python': False,
        'dependencies': False,
        'spacy': False,
        'data_files': False,
        'neo4j': False,
        'port': False
    }
    
    # Run all checks
    checks_passed['python'] = check_python_version()
    
    if checks_passed['python']:
        checks_passed['dependencies'] = check_dependencies()
        checks_passed['spacy'] = check_spacy_model()
        checks_passed['data_files'] = check_data_files()
        checks_passed['neo4j'] = check_neo4j_connection()
        checks_passed['port'] = check_port_available()
    
    # Print summary
    print_header("VALIDATION SUMMARY")
    
    status_symbols = {True: "✓", False: "✗"}
    
    for check_name, passed in checks_passed.items():
        status = status_symbols[passed]
        print(f"{status} {check_name.replace('_', ' ').title()}")
    
    # Suggest next steps
    suggest_next_steps(checks_passed)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup check cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)