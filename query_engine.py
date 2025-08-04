import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize paths and models
INDEX_PATH = "summary_index.faiss"
METADATA_PATH = "summary_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Number of most relevant summaries
TOP_K = 3

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

# Make sure number of embeddings matches number of metadata
assert index.ntotal == len(metadata), "Index and metadata size mismatch"

# Query Loop 
print("\nType your question. Type 'exit' to quit.\n")
while True:
    query = input("Query: ").strip()
    if query.lower() in {"exit"}:
        break

    # Embed query
    query_vec = model.encode([query]).astype("float32")

    # Search for best matches using distances D and indices I
    D, I = index.search(query_vec, TOP_K)

    print("\nTop Matches:")
    # Go through top K matches and output summaries, metadata for user
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
        entry = metadata[idx]
        summary = entry.get("summary", "[No summary]")
        meta_str = ", ".join(f"{k}: {v}" for k, v in entry.items() if k != "summary")
        print(f"\n[{rank}] {summary}")
        print(f"    ({meta_str})")
        print(f"    Distance: {dist:.4f}")
    print()
