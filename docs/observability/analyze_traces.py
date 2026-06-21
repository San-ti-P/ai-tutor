#!/usr/bin/env python3
"""Analyze Langfuse traces export -> emit metrics JSON for the deck."""
import json
import pandas as pd

CSV = "/home/santiago/workspaces/ai-tutor/docs/observability/traces.csv"
df = pd.read_csv(CSV)

df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce", utc=True)
df["latencyMs"] = pd.to_numeric(df["latencyMs"], errors="coerce")

def to_int(v):
    try:
        return int(float(v))
    except Exception:
        return 0

def parse_usage(x):
    try:
        d = json.loads(x)
        return to_int(d.get("input", 0)), to_int(d.get("output", 0)), to_int(d.get("total", 0))
    except Exception:
        return 0, 0, 0

usage = df["usageDetails"].apply(parse_usage)
df["tok_in"] = usage.apply(lambda t: t[0])
df["tok_out"] = usage.apply(lambda t: t[1])
df["tok_total"] = usage.apply(lambda t: t[2] if t[2] else t[0] + t[1])

gens = df[df["type"] == "GENERATION"]

# Tool usage
tool_rows = df[df["type"] == "TOOL"]
tool_counts = tool_rows["name"].value_counts().to_dict()

# Agent-level spans
agent_rows = df[df["type"] == "AGENT"]

# Pipeline operation counts (the meaningful named operations)
op_counts = df["name"].value_counts().head(25).to_dict()

# Latency for full LangGraph runs (end-to-end agent turns)
lg = df[df["name"] == "LangGraph"]["latencyMs"].dropna()

metrics = {
    "time_start": str(df["startTime"].min()),
    "time_end": str(df["startTime"].max()),
    "total_observations": int(len(df)),
    "unique_traces": int(df["traceId"].nunique()),
    "unique_sessions": int(df["sessionId"].nunique()),
    "type_breakdown": df["type"].value_counts().to_dict(),
    "errors": int((df["level"] == "ERROR").sum()),
    "success_rate_pct": round((1 - (df["level"] == "ERROR").sum() / len(df)) * 100, 2),
    "model": df["providedModelName"].dropna().unique().tolist(),
    "llm_generations": int(len(gens)),
    "tok_in": int(df["tok_in"].sum()),
    "tok_out": int(df["tok_out"].sum()),
    "tok_total": int(df["tok_total"].sum()),
    "total_cost_usd": float(pd.to_numeric(df["totalCost"], errors="coerce").sum()),
    "latency_gen_mean_ms": round(gens["latencyMs"].mean(), 0) if len(gens) else 0,
    "latency_gen_median_ms": round(gens["latencyMs"].median(), 0) if len(gens) else 0,
    "latency_langgraph_mean_ms": round(lg.mean(), 0) if len(lg) else 0,
    "latency_langgraph_median_ms": round(lg.median(), 0) if len(lg) else 0,
    "tool_counts": tool_counts,
    "op_counts": op_counts,
    # Key pipeline operation counts for narrative
    "n_retrieve_chunks": int((df["name"] == "retrieve_chunks").sum()),
    "n_rag_retrieve": int((df["name"] == "rag_retrieve").sum()),
    "n_rag_embed_store": int((df["name"] == "rag_embed_store").sum()),
    "n_validate_grounding": int((df["name"] == "validate_claim_grounding").sum()),
    "n_generate_exam": int((df["name"] == "generate_exam").sum()),
    "n_generate_exercise": int((df["name"] == "generate_exercise").sum()),
    "n_evaluate_answer": int((df["name"] == "evaluate_answer").sum()),
    "n_ingest_document": int((df["name"] == "ingest_document").sum()),
    "n_embeddings": int((df["type"] == "EMBEDDING").sum()),
    "n_retriever": int((df["type"] == "RETRIEVER").sum()),
}

with open("/home/santiago/workspaces/ai-tutor/docs/observability/trace_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
