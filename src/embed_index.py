from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

# Embedding model 
model = SentenceTransformer("all-MiniLM-L6-v2")

summaries: list[str] = []
metadata: list[dict] = []


def _ensure_datetime(s):
    return pd.to_datetime(s, errors="coerce")


def _month_name_from_int(m: int) -> str:
    return pd.Timestamp(year=2000, month=int(m), day=1).strftime("%B")

# Domain (daily granularity)
df_domain = pd.read_parquet(BASE_DIR / "data" / "domain_aggregates.parquet")

df_domain["date"] = _ensure_datetime(df_domain["date"])
if "year" not in df_domain.columns:
    df_domain["year"] = df_domain["date"].dt.year
if "month" not in df_domain.columns:
    df_domain["month"] = df_domain["date"].dt.month
if "month_name" not in df_domain.columns:
    df_domain["month_name"] = df_domain["date"].dt.strftime("%B")

def summarize_domain_daily(row: pd.Series) -> str:
    d = _ensure_datetime(row["date"])
    month_name = row["month_name"]
    year = int(row["year"])
    domain = str(row["domain"]).upper()
    avg_val = float(row["value"])
    tx = int(round(row["transaction_count"]))
    return (
        f"{month_name} {year} (on {d.strftime('%Y-%m-%d')}) — {domain} — "
        f"average value: {avg_val:.2f}; transactions: {tx} (transaction_count)."
    )

domain_daily_summaries = df_domain.apply(summarize_domain_daily, axis=1).tolist()
summaries.extend(domain_daily_summaries)
metadata.extend(
    {
        "granularity": "domain_daily",
        "type": "domain_daily",
        "date": str(row["date"]),
        "year": int(row["year"]),
        "month": int(row["month"]),
        "month_name": row["month_name"],
        "domain": str(row["domain"]).upper(),
        "summary": summary,
    }
    for row, summary in zip(df_domain.to_dict(orient="records"), domain_daily_summaries)
)

# DOMAIN (monthly granularity)
df_domain_monthly = pd.read_parquet(BASE_DIR / "data" / "domain_monthly_aggregates.parquet")

# force month name
if "month_name" not in df_domain_monthly.columns:
    df_domain_monthly["month_name"] = df_domain_monthly["month"].apply(_month_name_from_int)

def summarize_domain_monthly(row: pd.Series) -> str:
    month_name = row["month_name"]
    year = int(row["year"])
    domain = str(row["domain"]).upper()
    avg_val = float(row["value"])
    # monthly aggregate uses sum of transaction_count
    tx_sum = int(round(row["transaction_count"]))
    return (
        f"{month_name} {year} — {domain} — "
        f"sum of transaction_count: {tx_sum}; average value: {avg_val:.2f}."
    )

domain_monthly_summaries = df_domain_monthly.apply(summarize_domain_monthly, axis=1).tolist()
summaries.extend(domain_monthly_summaries)
metadata.extend(
    {
        "granularity": "domain_monthly",
        "type": "domain_monthly",
        "year": int(row["year"]),
        "month": int(row["month"]),
        "month_name": row["month_name"],
        "domain": str(row["domain"]).upper(),
        "summary": summary,
    }
    for row, summary in zip(df_domain_monthly.to_dict(orient="records"), domain_monthly_summaries)
)

# City
df_city = pd.read_parquet(BASE_DIR / "data" / "city_aggregates.parquet")

def summarize_city(row: pd.Series) -> str:
    loc = str(row["location"])
    val_mean = float(row["value_mean"])
    val_sum = float(row["value_sum"])
    return (
        f"In {loc}, average value: {val_mean:.2f}; total value: {val_sum:.2f}."
    )

city_summaries = df_city.apply(summarize_city, axis=1).tolist()
summaries.extend(city_summaries)
metadata.extend(
    {
        "granularity": "city",
        "type": "city",
        "location": str(row["location"]),
        "summary": summary,
    }
    for row, summary in zip(df_city.to_dict(orient="records"), city_summaries)
)

df_daily = pd.read_parquet(BASE_DIR / "data" / "daily_totals.parquet")
df_daily["date"] = _ensure_datetime(df_daily["date"])
df_daily["year"] = df_daily["date"].dt.year
df_daily["month"] = df_daily["date"].dt.month
df_daily["month_name"] = df_daily["date"].dt.strftime("%B")

def summarize_daily_total(row: pd.Series) -> str:
    d = _ensure_datetime(row["date"])
    month_name = row["month_name"]
    year = int(row["year"])
    val = float(row["value"])
    tx = int(round(row["transaction_count"]))
    return (
        f"{month_name} {year} (on {d.strftime('%Y-%m-%d')}) — "
        f"total value: {val:.2f}; transactions: {tx} (transaction_count)."
    )

daily_total_summaries = df_daily.apply(summarize_daily_total, axis=1).tolist()
summaries.extend(daily_total_summaries)
metadata.extend(
    {
        "granularity": "daily_total",
        "type": "daily",
        "date": str(row["date"]),
        "year": int(row["year"]),
        "month": int(row["month"]),
        "month_name": row["month_name"],
        "summary": summary,
    }
    for row, summary in zip(df_daily.to_dict(orient="records"), daily_total_summaries)
)

# Minimal enrichment for better retrieval
for i, meta in enumerate(metadata):
    parts = []
    if "domain" in meta:
        parts.append(str(meta["domain"]))
    if "year" in meta:
        parts.append(str(meta["year"]))
    if "month_name" in meta:
        parts.append(str(meta["month_name"]))
    if "value" in meta:
        parts.append(f"value {meta['value']}")
    if "transaction_count" in meta:
        parts.append(f"transaction_count {meta['transaction_count']}")
    # append metadata tokens to the summary text
    summaries[i] = summaries[i] + " · " + " · ".join(parts)

# Build FAISS index
print(f"Encoding {len(summaries)} summaries...")
embeddings = model.encode(summaries, show_progress_bar=True)
embeddings = np.asarray(embeddings, dtype="float32")

# Cosine similarity (normalize + inner product)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Save
faiss.write_index(index, str(BASE_DIR / "data" / "summary_index.faiss"))
with open(BASE_DIR / "data" / "summary_metadata.json", "w") as f:
    json.dump(metadata, f)

print("FAISS index and metadata saved.")
