import faiss
import json
import ollama
from sentence_transformers import SentenceTransformer

"""
How to use:

1. Install required Python packages:
       pip install -r requirements.txt
2. Install Ollama (https://ollama.com/download)
3. Start Ollama server: ollama serve
4. (Optional but recommended) Create and activate a Python virtual environment:
       python -m venv venv
       source venv/bin/activate        # Mac/Linux
       venv\Scripts\activate           # Windows
5. Run: python query_engine.py
6. Ask questions that match the summaries for best results

- Requires summary_index.faiss and summary_metadata.json in the same folder
- Ollama server must be running first
"""

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
    # Go through top K matches and output summaries, metadata 
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
        entry = metadata[idx]
        summary = entry.get("summary", "[No summary]")
        meta_str = ", ".join(f"{k}: {v}" for k, v in entry.items() if k != "summary")
        print(f"\n[{rank}] {summary}")
        print(f"    ({meta_str})")
        print(f"    Distance: {dist:.4f}")
    
    # Build context from top matches
    context = ""
    for idx in I[0]:
        entry = metadata[idx]
        summary = entry.get("summary", "[No summary]")
        context += summary + "\n"

    # Create RAG prompt
    prompt = f"""Use the following data to answer the question as clearly as possible.

{context}

Question: {query}
Answer:"""

    # Call local LLM via Ollama
    response = ollama.chat(
        model='phi3',
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nGenerated Answer:")
    print(response['message']['content'])
    print()
