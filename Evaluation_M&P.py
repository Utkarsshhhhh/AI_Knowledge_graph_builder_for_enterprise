import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime

# Load triples
print("Loading knowledge base...")
with open(r"C:\Users\ashwa\OneDrive\Desktop\AI_Graph\Internship\knowledge_triples.json", 'r', encoding='utf-8') as f:
    triples = json.load(f)

# Create texts
texts = [f"{t['subject']} {t['predicate']} {t['object']}" for t in triples]

# Load model and build index
print("Building search index...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"✓ Indexed {index.ntotal} entries\n")

# Evaluation Dataset
# Format: (query, [relevant_indices])
# You need to manually create this based on your data
EVAL_DATASET = [
    ("president", [0, 1, 2]),           # Indices of triples about presidents
    ("technology", [5, 6, 7, 8]),       # Indices of triples about technology
    ("company CEO", [10, 11, 12]),      # Indices of triples about CEOs
    ("capital city", [15, 16]),         # Indices of triples about capitals
    ("artificial intelligence", [20, 21, 22, 23]),  # AI-related triples
]

# Search function
def search(query, k=5):
    """Perform search and return indices"""
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]

# Evaluation Metrics
def calculate_precision(retrieved, relevant):
    """Precision = (Retrieved ∩ Relevant) / Retrieved"""
    if len(retrieved) == 0:
        return 0.0
    tp = len(set(retrieved).intersection(set(relevant)))
    return tp / len(retrieved)

def calculate_recall(retrieved, relevant):
    """Recall = (Retrieved ∩ Relevant) / Relevant"""
    if len(relevant) == 0:
        return 0.0
    tp = len(set(retrieved).intersection(set(relevant)))
    return tp / len(relevant)

def calculate_f1(precision, recall):
    """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_map(retrieved, relevant):
    """Mean Average Precision"""
    if len(relevant) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(retrieved):
        if item in relevant:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / len(relevant)

def calculate_mrr(retrieved, relevant):
    """Mean Reciprocal Rank"""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0

def calculate_ndcg(retrieved, relevant, k):
    """Normalized Discounted Cumulative Gain"""
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    
    return dcg / idcg if idcg > 0 else 0.0

# Run Evaluation
def evaluate_system(dataset, k_values=[1, 3, 5, 10]):
    """Comprehensive evaluation"""
    
    print("="*70)
    print("EVALUATION REPORT - RAG SYSTEM")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Test Queries: {len(dataset)}")
    print(f"Knowledge Base Size: {len(texts)}")
    print("="*70)
    
    results = {k: {'precision': [], 'recall': [], 'f1': [], 'map': [], 'mrr': [], 'ndcg': []} 
               for k in k_values}
    
    # Evaluate each query
    print("\nPer-Query Results:")
    print("-"*70)
    
    for query, relevant in dataset:
        print(f"\nQuery: '{query}'")
        print(f"Relevant items: {len(relevant)}")
        
        for k in k_values:
            retrieved, distances = search(query, k)
            
            precision = calculate_precision(retrieved, relevant)
            recall = calculate_recall(retrieved, relevant)
            f1 = calculate_f1(precision, recall)
            map_score = calculate_map(retrieved, relevant)
            mrr_score = calculate_mrr(retrieved, relevant)
            ndcg_score = calculate_ndcg(retrieved, relevant, k)
            
            results[k]['precision'].append(precision)
            results[k]['recall'].append(recall)
            results[k]['f1'].append(f1)
            results[k]['map'].append(map_score)
            results[k]['mrr'].append(mrr_score)
            results[k]['ndcg'].append(ndcg_score)
        
        # Show results for k=5
        k = 5
        retrieved, _ = search(query, k)
        print(f"  Retrieved@{k}: {list(retrieved)}")
        print(f"  Precision@{k}: {results[k]['precision'][-1]:.3f}")
        print(f"  Recall@{k}: {results[k]['recall'][-1]:.3f}")
        print(f"  F1@{k}: {results[k]['f1'][-1]:.3f}")
    
    # Overall metrics
    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    
    for k in k_values:
        avg_precision = np.mean(results[k]['precision'])
        avg_recall = np.mean(results[k]['recall'])
        avg_f1 = np.mean(results[k]['f1'])
        avg_map = np.mean(results[k]['map'])
        avg_mrr = np.mean(results[k]['mrr'])
        avg_ndcg = np.mean(results[k]['ndcg'])
        
        print(f"\n--- Metrics @ K={k} ---")
        print(f"  Precision: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
        print(f"  Recall:    {avg_recall:.3f} ({avg_recall*100:.1f}%)")
        print(f"  F1-Score:  {avg_f1:.3f} ({avg_f1*100:.1f}%)")
        print(f"  MAP:       {avg_map:.3f} ({avg_map*100:.1f}%)")
        print(f"  MRR:       {avg_mrr:.3f} ({avg_mrr*100:.1f}%)")
        print(f"  NDCG:      {avg_ndcg:.3f} ({avg_ndcg*100:.1f}%)")
    
    print("\n" + "="*70)
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(dataset),
        'knowledge_base_size': len(texts),
        'metrics': {}
    }
    
    for k in k_values:
        report['metrics'][f'k_{k}'] = {
            'precision': float(np.mean(results[k]['precision'])),
            'recall': float(np.mean(results[k]['recall'])),
            'f1': float(np.mean(results[k]['f1'])),
            'map': float(np.mean(results[k]['map'])),
            'mrr': float(np.mean(results[k]['mrr'])),
            'ndcg': float(np.mean(results[k]['ndcg']))
        }
    
    with open('evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✓ Report saved to evaluation_report.json")
    
    return results

# Metric Explanations
def explain_metrics():
    """Print metric explanations"""
    print("\n" + "="*70)
    print("METRIC EXPLANATIONS")
    print("="*70)
    
    explanations = {
        "Precision@K": "What % of retrieved results are relevant? (Higher is better)",
        "Recall@K": "What % of relevant items were retrieved? (Higher is better)",
        "F1-Score": "Harmonic mean of Precision and Recall (Higher is better)",
        "MAP": "Mean Average Precision - ranking quality (Higher is better)",
        "MRR": "Mean Reciprocal Rank - position of first relevant result (Higher is better)",
        "NDCG": "Normalized Discounted Cumulative Gain - ranking quality (Higher is better)"
    }
    
    for metric, explanation in explanations.items():
        print(f"\n{metric}:")
        print(f"  {explanation}")
    
    print("\n" + "="*70)

# Run evaluation
explain_metrics()
results = evaluate_system(EVAL_DATASET, k_values=[1, 3, 5, 10])

print("\n✓ Evaluation Complete!")
print("\nNOTE: Update EVAL_DATASET with your actual ground truth data")
print("      Format: (query, [relevant_triple_indices])")