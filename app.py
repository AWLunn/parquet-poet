import re
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# RAG imports
try:
    from src.query_engine import load_rag_components, query_rag  # type: ignore
except ImportError:
    from query_engine import load_rag_components, query_rag  # type: ignore

# LLM chart spec (used, then normalized)
try:
    from src.query_engine import llm_chart_spec  # type: ignore
except ImportError:
    from query_engine import llm_chart_spec  # type: ignore

st.set_page_config(
    page_title="Ask My Data",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Ask My Data 🤖")
st.write("Ask questions about the bank dataset using natural language.")

# Normalization rules for LLM output
SUPPORTED_BY = {"year", "quarter", "month", "week", "day"}
METRIC_ALIASES = {
    "count": "transaction_count",
    "txn": "transaction_count",
    "txns": "transaction_count",
    "transactions": "transaction_count",
    "transaction_count": "transaction_count",
    "value": "value",
    "amount": "value",
}

def normalize_spec(raw_spec: dict | None, domains: list[str], df_domain: pd.DataFrame) -> dict:
    """Clamp LLM output to valid domain, metric, and by. Provide safe defaults."""
    spec = {} if raw_spec is None else dict(raw_spec)

    # domain or domains[0]
    dom = (spec.get("domains") or [spec.get("domain")])[0] if ("domains" in spec or "domain" in spec) else ""
    if dom:
        dom_lower = str(dom).lower()
        exact = [d for d in domains if d.lower() == dom_lower]
        contains = [d for d in domains if dom_lower in d.lower()]
        dom = (exact + contains + [str(dom).upper()])[0]
    else:
        dom = domains[0] if domains else ""

    # metric with alias map
    metric = str(spec.get("metric") or "value").strip().lower().replace(" ", "_")
    metric = METRIC_ALIASES.get(metric, metric)
    if metric not in df_domain.columns:
        metric = "value"

    # time unit
    by_raw = str(spec.get("by") or "month").strip().lower()
    by_map = {"yr": "year", "annual": "year", "weekly": "week", "daily": "day", "q": "quarter", "quarterly": "quarter"}
    by = by_map.get(by_raw, by_raw)
    if by not in SUPPORTED_BY:
        by = "month"

    return {"domain": dom, "metric": metric, "by": by}

# Data
@st.cache_data
def load_data():
    base = Path(__file__).parent / "data"
    files = {
        "domain": base / "domain_aggregates.parquet",
        "city":   base / "city_aggregates.parquet",
        "daily":  base / "daily_totals.parquet",
    }
    dfs: dict[str, pd.DataFrame] = {}
    for key, path in files.items():
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "domain" in df.columns:
            df["domain"] = (
                df["domain"].astype(str).str.strip().str.upper()
                .str.replace(r"\s+", " ", regex=True)
            )
        if "city" in df.columns:
            df["city"] = (
                df["city"].astype(str).str.strip().str.title()
                .str.replace(r"\s+", " ", regex=True)
            )
        dfs[key] = df
    return dfs

dfs = load_data()
df = dfs.get("domain")
if df is None or df.empty:
    st.error("Missing domain_aggregates.parquet in ./data")
    st.stop()


# RAG
@st.cache_resource
def _load_rag():
    return load_rag_components()

rag_model, index, metadata = _load_rag()
if rag_model is None or index is None or metadata is None:
    st.error("RAG model components could not be loaded. Check files and imports.")
    st.stop()

with st.expander("Dataset Overview"):
    st.dataframe(df.head(10), use_container_width=True)


# Grain auto selection to avoid single buckets
def _best_grain(dates: pd.Series, requested: str) -> str:
    requested = requested.lower()
    def uniq(period: str) -> int:
        return dates.dt.to_period(period).nunique()

    if requested == "year":
        return "year" if uniq("Y") >= 2 else "month"
    if requested == "quarter":
        return "quarter" if uniq("Q") >= 2 else "month"
    if requested == "month":
        return "month" if uniq("M") >= 2 else "week"
    if requested == "week":
        return "week" if uniq("W-MON") >= 2 else "day"
    return requested  # day stays day

# Charting
def make_chart(df_domain: pd.DataFrame, domain: str, by: str = "month", metric: str = "value"):
    d = df_domain[df_domain["domain"].astype(str).str.upper() == domain.upper()].copy()
    if d.empty:
        return None, f"No rows for '{domain}'."

    # choose a grain that yields at least 2 buckets when possible
    by = _best_grain(d["date"], by)

    if by == "year":
        d["period"] = d["date"].dt.to_period("Y").dt.to_timestamp()
    elif by == "quarter":
        d["period"] = d["date"].dt.to_period("Q").dt.to_timestamp()
    elif by == "month":
        d["period"] = d["date"].dt.to_period("M").dt.to_timestamp()
    elif by == "week":
        d["period"] = d["date"].dt.to_period("W-MON").dt.start_time
    elif by == "day":
        d["period"] = d["date"]
    else:
        return None, f"Unsupported time aggregation: {by}"

    if metric not in d.columns:
        return None, f"Unknown metric '{metric}'."

    agg = d.groupby("period", as_index=False)[metric].sum()

    chart = (
        alt.Chart(agg)
        .mark_line(point=True)
        .encode(
            x=alt.X("period:T", title=by.title()),
            y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
            tooltip=["period:T", alt.Tooltip(f"{metric}:Q", format=",.2f")],
        )
        .properties(title=f"{domain.title()} — {metric.replace('_',' ').title()} by {by.title()}")
        .interactive()
    )
    return chart, None


# Chat handler
prompt = st.chat_input("Type a question, or: graph <DOMAIN> [metric] by [|quarter|month|day]")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if prompt.lower().startswith("graph"):
                try:
                    domains = sorted(df["domain"].astype(str).unique().tolist())
                except Exception:
                    domains = []

                raw_spec = None
                try:
                    if domains:
                        raw_spec = llm_chart_spec(prompt, domains)  # may return quarter or aliases
                except Exception:
                    raw_spec = None

                spec = normalize_spec(raw_spec, domains, df)
                domain = spec["domain"]
                by = spec["by"]
                metric = spec["metric"]

                chart, err = make_chart(df, domain, by=by, metric=metric)
                if err:
                    st.warning(err)
                else:
                    st.altair_chart(chart, use_container_width=True)
            else:
            # tighten LLM answers: use real domains, be concise, no tables/guesses
                try:
                    domain_list = sorted(df["domain"].astype(str).unique().tolist())
                except Exception:
                    domain_list = []

                guardrails = f"""
                Use ONLY these domain names exactly: {", ".join(domain_list)}.
                Do not invent categories or domains.
                If asked for a ranking (e.g., "top N"), reply as a short numbered list with exact domain names and metric values.
                If asked for totals, reply with one concise sentence containing the exact number(s).
                No tables. No long explanations. No guesses.
                If context is insufficient, reply exactly: "Not enough information."
                Use metric names 'value' and 'transaction_count'. Format numbers with thousands separators.
                """

                safe_prompt = f"{prompt}\n\n{guardrails}".strip()
                answer = query_rag(safe_prompt, rag_model, index, metadata).strip()
                st.markdown(answer)
