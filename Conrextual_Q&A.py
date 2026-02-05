
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')


class ContextualQA:
    """Semantic question-answering system using FAISS"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the QA system
        
        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.context = []
        self.embeddings = None
        
    def load_triples(self, file_path: str) -> pd.DataFrame:
        """
        Load triples from JSON or CSV file
        
        Args:
            file_path: Path to JSON or CSV file
            
        Returns:
            DataFrame with columns [head, relation, tail]
        """
        print(f"Loading knowledge base from: {file_path}")
        
        try:
            # Determine file type
            if file_path.endswith('.json'):
                return self._load_from_json(file_path)
            else:
                return self._load_from_csv(file_path)
                
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return pd.DataFrame()
    
    def _load_from_json(self, json_path: str) -> pd.DataFrame:
        """Load triples from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of triples
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Dictionary with triples
                if 'triples' in data:
                    df = pd.DataFrame(data['triples'])
                else:
                    # Assume the dict itself contains the data
                    df = pd.DataFrame([data])
            else:
                print(f"✗ Unexpected JSON structure")
                return pd.DataFrame()
            
            # Normalize column names
            df = self._normalize_columns(df)
            
            print(f"✓ Loaded {len(df)} triples from JSON")
            return df
            
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON format: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"✗ Error reading JSON: {e}")
            return pd.DataFrame()
    
    def _load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load triples from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            df = self._normalize_columns(df)
            print(f"✓ Loaded {len(df)} triples from CSV")
            return df
        except Exception as e:
            print(f"✗ Error reading CSV: {e}")
            return pd.DataFrame()
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to standard format [head, relation, tail]
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized columns
        """
        # Handle different column naming conventions
        column_mappings = [
            # Standard formats
            {'subject': 'head', 'predicate': 'relation', 'object': 'tail'},
            {'Subject': 'head', 'Predicate': 'relation', 'Object': 'tail'},
            {'s': 'head', 'p': 'relation', 'o': 'tail'},
            {'source': 'head', 'relation': 'relation', 'target': 'tail'},
            {'from': 'head', 'type': 'relation', 'to': 'tail'},
            {'entity1': 'head', 'relation': 'relation', 'entity2': 'tail'},
            {'h': 'head', 'r': 'relation', 't': 'tail'},
        ]
        
        # Try each mapping
        for mapping in column_mappings:
            if all(col in df.columns for col in mapping.keys()):
                df = df.rename(columns=mapping)
                break
        
        # If standard columns not found, use first 3 columns
        if not all(col in df.columns for col in ['head', 'relation', 'tail']):
            if len(df.columns) >= 3:
                print(f"⚠ Using first 3 columns: {df.columns[0]}, {df.columns[1]}, {df.columns[2]}")
                df.columns = ['head', 'relation', 'tail'] + list(df.columns[3:])
            else:
                print(f"✗ Not enough columns! Found: {df.columns.tolist()}")
                return pd.DataFrame()
        
        # Clean data
        df = df.dropna(subset=['head', 'relation', 'tail'])
        df['head'] = df['head'].astype(str).str.strip()
        df['relation'] = df['relation'].astype(str).str.strip()
        df['tail'] = df['tail'].astype(str).str.strip()
        
        # Remove empty strings
        df = df[(df['head'] != '') & (df['relation'] != '') & (df['tail'] != '')]
        
        return df
    
    def create_context(self, df: pd.DataFrame) -> List[str]:
        """
        Create natural language context from triples
        
        Args:
            df: DataFrame with columns [head, relation, tail]
            
        Returns:
            List of context sentences
        """
        print("Creating context...")
        
        if df.empty:
            print("✗ No data to create context from")
            return []
        
        context = []
        
        for _, row in df.iterrows():
            # Create natural language sentences from triples
            head = str(row['head']).strip()
            relation = str(row['relation']).strip()
            tail = str(row['tail']).strip()
            
            # Skip empty values
            if not head or not tail or not relation:
                continue
            
            # Format relation for readability
            relation_formatted = relation.replace('_', ' ').replace('-', ' ').lower()
            
            # Create sentence
            sentence = f"{head} {relation_formatted} {tail}"
            context.append(sentence)
        
        self.context = context
        print(f"✓ Created {len(context)} context entries")
        
        return context
    
    def load_model(self):
        """Load the sentence transformer model"""
        print("Loading embedding model...")
        try:
            self.model = SentenceTransformer(self.model_name)
            print("✓ Model loaded")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def generate_embeddings(self, batch_size: int = 32):
        """
        Generate embeddings for all context entries
        
        Args:
            batch_size: Number of sentences to process at once
        """
        if not self.context:
            print("✗ No context to embed. Please create context first.")
            return
        
        if self.model is None:
            print("✗ Model not loaded. Please load model first.")
            return
        
        print("Generating embeddings...")
        try:
            # Generate embeddings in batches
            self.embeddings = self.model.encode(
                self.context,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            print(f"✓ Generated embeddings with shape: {self.embeddings.shape}")
            
        except Exception as e:
            print(f"✗ Error generating embeddings: {e}")
            raise
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if self.embeddings is None or len(self.embeddings) == 0:
            print("✗ No embeddings available. Please generate embeddings first.")
            return
        
        print("Building FAISS index...")
        try:
            # Get embedding dimension
            dimension = self.embeddings.shape[1]
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            
            # Create FAISS index (Inner Product = Cosine Similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
            
            # Add embeddings to index
            self.index.add(self.embeddings)
            
            print(f"✓ Built FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            print(f"✗ Error building FAISS index: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for most relevant context entries
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (context, score) tuples
        """
        if self.index is None:
            print("✗ FAISS index not built. Please build index first.")
            return []
        
        if self.model is None:
            print("✗ Model not loaded. Please load model first.")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Normalize query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.context):
                results.append((self.context[idx], float(score)))
        
        return results
    
    def answer_question(self, question: str, top_k: int = 5) -> str:
        """
        Answer a question using retrieved context
        
        Args:
            question: User question
            top_k: Number of context entries to retrieve
            
        Returns:
            Formatted answer with context
        """
        results = self.search(question, top_k)
        
        if not results:
            return "No relevant information found."
        
        # Format answer
        answer = f"Question: {question}\n\n"
        answer += "Relevant Information:\n"
        answer += "-" * 60 + "\n"
        
        for i, (context, score) in enumerate(results, 1):
            answer += f"{i}. {context} (relevance: {score:.3f})\n"
        
        return answer


def main():
    """Main execution flow"""
    
    # Configuration
    FILE_PATH = "Internship\entity_relation_entity_triples.json"  # Can be .json or .csv
    MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast and efficient model
    
    # Initialize QA system
    qa = ContextualQA(model_name=MODEL_NAME)
    
    # Step 1: Load triples
    df = qa.load_triples(FILE_PATH)
    
    if df.empty:
        print("\n✗ Failed to load data. Exiting...")
        print("\n💡 Tips:")
        print("  1. Check that the file exists")
        print("  2. For JSON: should be a list of objects with triple fields")
        print("  3. For CSV: should have subject/predicate/object or head/relation/tail columns")
        return
    
    # Show sample data
    print(f"\n📊 Sample data (first 3 triples):")
    for i, row in df.head(3).iterrows():
        print(f"  {i+1}. {row['head']} --[{row['relation']}]--> {row['tail']}")
    
    # Step 2: Create context
    context = qa.create_context(df)
    
    if not context:
        print("\n✗ Failed to create context. Exiting...")
        print("\nDebugging info:")
        print(f"  - DataFrame shape: {df.shape}")
        print(f"  - DataFrame columns: {df.columns.tolist()}")
        print(f"  - First few rows:")
        print(df.head())
        return
    
    # Step 3: Load model
    qa.load_model()
    
    # Step 4: Generate embeddings
    qa.generate_embeddings()
    
    if qa.embeddings is None or len(qa.embeddings) == 0:
        print("\n✗ Failed to generate embeddings. Exiting...")
        return
    
    # Step 5: Build FAISS index
    qa.build_faiss_index()
    
    if qa.index is None:
        print("\n✗ Failed to build FAISS index. Exiting...")
        return
    
    print("\n" + "="*60)
    print("🎯 CONTEXTUAL Q&A SYSTEM READY")
    print("="*60)
    print(f"Knowledge base: {len(qa.context)} facts")
    print(f"Embedding model: {MODEL_NAME}")
    print("Type 'quit' or 'exit' to stop\n")
    
    # Interactive Q&A loop
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Get answer
            answer = qa.answer_question(question, top_k=5)
            print("\n" + answer)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()