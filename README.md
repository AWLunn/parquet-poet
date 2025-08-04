# Ask My Data (parquet poet)

Ask My Data is a Retrieval-Augmented Generation (RAG) system that allows users to query structured datasets using natural language. It combines data cleaning, embedding, vector indexing, and large language model prompting to generate relevant, context-aware responses.

## Pipeline Overview

1. **Preprocessing**  
   Input data is cleaned and aggregated using pandas. Future versions may support PySpark and S3-scale datasets.

2. **Embedding**  
   Aggregated summaries are embedded using sentence-transformers.

3. **Indexing**  
   Embeddings are stored in a FAISS index alongside metadata.

4. **Querying**  
   Natural language queries are embedded and matched against the index. The top results are passed to an LLM (e.g., GPT-4) to generate a response.

## Status

- Preprocessing pipeline complete
- Embedding and FAISS indexing in progress
- Query engine and interface to follow

## Notes

This project is not licensed for reuse or redistribution.
