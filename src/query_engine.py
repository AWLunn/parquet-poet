from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths work whether this file is in src/ or root
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / "data") if (HERE / "data").exists() else (HERE.parent / "data")
INDEX_PATH = str(DATA_DIR / "summary_index.faiss")   # faiss wants str
METADATA_PATH = DATA_DIR / "summary_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

# Optional Ollama
try:
    import ollama
    OLLAMA_MODEL = "phi3:mini"
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

def load_rag_components():
    """Load SentenceTransformer, FAISS index, and metadata."""
    try:
        model = SentenceTransformer(MODEL_NAME)
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        assert index.ntotal == len(metadata), "Index and metadata size mismatch"
        return model, index, metadata
    except FileNotFoundError as e:
        print(f"Missing RAG files: {e}")
        return None, None, None

def query_rag(query: str, model, index, metadata):
    """Search FAISS and, if available, ask Ollama to answer using the top matches."""
    if model is None or index is None or metadata is None:
        return "RAG model is not loaded correctly."

    # Normalize query for cosine similarity
    qv = SentenceTransformer(MODEL_NAME).encode([query]).astype("float32")
    faiss.normalize_L2(qv)  # ensure cosine similarity matches build step
    D, I = index.search(qv, TOP_K)

    hits = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
        entry = metadata[idx]
        summary = entry.get("summary", "[No summary]")
        meta_str = ", ".join(f"{k}: {v}" for k, v in entry.items() if k != "summary")
        # Label score correctly and show enriched tokens 
        hits.append(
            f"[{rank}] {summary}\n({meta_str})\nSimilarity: {dist:.4f}"
        )

    if not HAS_OLLAMA:
        return "Top Matches from RAG System:\n\n" + "\n\n".join(hits)

    ctx = "\n\n".join(hits)
    prompt = (
        "Use the context to answer the question. If unclear, state the best inference.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )
    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        answer = resp["message"]["content"].strip()
        return answer + "\n\n---\nTop matches used:\n" + "\n\n".join(hits)
    except Exception:
        return "(FAISS-only mode — Ollama offline)\n\nTop Matches from RAG System:\n\n" + "\n\n".join(hits)

ALLOWED_BY = {"day","month","quarter"}
ALLOWED_METRIC = {"value","transaction_count"}

def _ollama_ready() -> bool:
    """Quickly check if Ollama model responds."""
    if not HAS_OLLAMA:
        return False
    try:
        ollama.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":"ping"}], options={"num_predict":1})
        return True
    except Exception:
        return False

def llm_chart_spec(user_text: str, domains: list[str]) -> dict | None:
    """Ask Ollama for a tiny JSON spec and validate it. Returns dict or None."""
    if not _ollama_ready():
        return None

    domain_list = ", ".join(sorted({d.upper() for d in domains})[:100])
    prompt = f"""
Return ONLY JSON. No prose. No code fences. Keys: domain, by, metric.

Rules:
- domain must be one of: {domain_list}
- by must be one of: day, month, quarter
- metric must be one of: value, transaction_count

User request: {user_text}

Example:
{{"domain":"MEDICAL","by":"month","metric":"value"}}
"""

    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":prompt}],
                           options={"temperature":0})
        txt = resp["message"]["content"].strip()

        # hard strip any fences the model might add
        if txt.startswith("```"):
            txt = txt.strip("`")
            if "\n" in txt:
                txt = txt.split("\n", 1)[1].strip()

        spec = json.loads(txt)

        # sanitize and validate
        spec = {
            "domain": str(spec.get("domain","")).upper()[:64],
            "by": str(spec.get("by","month")).lower(),
            "metric": str(spec.get("metric","value")).lower(),
        }
        if spec["by"] not in ALLOWED_BY: spec["by"] = "month"
        if spec["metric"] not in ALLOWED_METRIC: spec["metric"] = "value"
        if spec["domain"] not in {d.upper() for d in domains}:
            return None

        return spec
    except Exception:
        return None
