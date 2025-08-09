import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize paths and models
INDEX_PATH = "./data/summary_index.faiss"
METADATA_PATH = "./data/summary_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Number of most relevant summaries
TOP_K = 3

def load_rag_components():
    """ 
    Loads the SentenceTransformer, FAISS Index, and metadata
    """
    try:
        # Load model
        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)
        
        # Load FAISS
        print("Loading FAISS index...")
        index = faiss.read_index(INDEX_PATH)
        
        # Load metadata 
        print("Loading metadata...")
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
            
        assert index.ntotal == len(metadata), "Index and metadata size mismatch"
        
        return model, index, metadata
    except FileNotFoundError as e:
        print(f"Error: Require files not found. Details: {e}")
        return None, None, None


def query_rag(query, model, index, metadata):
    """
    Queries the RAG system with the given query.
    """
    if model is None or index is None or metadata is None:
        return "RAG model is not loaded correctly."

    query_vec = model.encode([query]).astype("float32")
    D, I = index.search(query_vec, TOP_K)

    response_parts = ["Top Matches from RAG System:"]
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
        entry = metadata[idx]
        summary = entry.get("summary", "[No summary]")
        meta_str = ", ".join(f"{k}: {v}" for k, v in entry.items() if k != "summary")
        response_parts.append(f"\n[{rank}] {summary}")
        response_parts.append(f"    ({meta_str})")
        response_parts.append(f"    Distance: {dist:.4f}")
    
    return "\n".join(response_parts)
