import pandas as pd
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer

# Pre-trained model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

summaries = []
metadata = []

# Domain aggregates
df_domain = pd.read_parquet("domain_aggregates.parquet")

# Create semantic handles
def summarize_domain(row):
    return (
        f"On {row['date'].strftime('%Y-%m-%d')}, the {row['domain']} domain had an average value of "
        f"{row['value']:.2f} across {int(row['transaction_count'])} transactions."
    )

domain_summaries = df_domain.apply(summarize_domain, axis=1).tolist()

# Prepare data for FAISS and JSON
summaries.extend(domain_summaries)
metadata.extend([
    {
        "type": "domain",
        "date": str(row["date"]),
        "domain": row["domain"],
        "summary": summary
    }
    for row, summary in zip(df_domain.to_dict(orient="records"), domain_summaries)
])


# City aggregates
df_city = pd.read_parquet("city_aggregates.parquet")

# Create semantic handles
def summarize_city(row):
    return (
        f"In {row['location']}, the average transaction value was {row['value_mean']:.2f} "
        f"and the total value was {row['value_sum']:.2f}."
    )

# Prepare data for FAISS and JSON
city_summaries = df_city.apply(summarize_city, axis=1).tolist()
summaries.extend(city_summaries)
metadata.extend([
    {
        "type": "city",
        "location": row["location"],
        "summary": summary
    }
    for row, summary in zip(df_city.to_dict(orient="records"), city_summaries)
])


# Daily total
df_daily = pd.read_parquet("daily_totals.parquet")

# Create semantic handles
def summarize_daily(row):
    return (
        f"On {row['date'].strftime('%Y-%m-%d')}, the total value of all transactions was {row['value']:.2f} "
        f"across {int(row['transaction_count'])} transactions."
    )

# Prepare data for FAISS and JSON
daily_summaries = df_daily.apply(summarize_daily, axis=1).tolist()
summaries.extend(daily_summaries)
metadata.extend([
    {
        "type": "daily",
        "date": str(row["date"]),
        "summary": summary
    }
    for row, summary in zip(df_daily.to_dict(orient="records"), daily_summaries)
])


# Encoding and FAISS prep
print(f"Encoding {len(summaries)} summaries...")
embeddings = model.encode(summaries, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index based on Euclidean distance (L2)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save to disk
faiss.write_index(index, "summary_index.faiss")

# Store metadata in JSON
with open("summary_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("FAISS index and metadata saved.")
