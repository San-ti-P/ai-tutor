#!/usr/bin/env python3
"""Reconstruct a REAL exam-generation trace tree from the Langfuse CSV export
and render it as a Langfuse-style waterfall image."""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CSV = "/home/santiago/workspaces/ai-tutor/docs/observability/traces.csv"
OUT = "/home/santiago/workspaces/ai-tutor/docs/presentation/assets/exam_trace.png"

df = pd.read_csv(CSV)
df["latencyMs"] = pd.to_numeric(df["latencyMs"], errors="coerce")

# Find the traceId of a rich generate_exam run (most observations under it)
exam_traces = df[df["name"] == "generate_exam"]["traceId"].value_counts()
# pick the trace whose subtree has the most rows AND contains generate_questions
best = None
for tid in exam_traces.index:
    sub = df[df["traceId"] == tid]
    if (sub["name"] == "generate_questions").any() and (sub["name"] == "retrieve_chunks").any():
        best = tid
        break
if best is None:
    best = exam_traces.index[0]

sub = df[df["traceId"] == best].copy()

# Build id -> row map and children map
by_id = {r["id"]: r for _, r in sub.iterrows()}
children = {}
roots = []
for _, r in sub.iterrows():
    pid = r["parentObservationId"]
    if pd.isna(pid) or pid not in by_id:
        roots.append(r["id"])
    else:
        children.setdefault(pid, []).append(r["id"])

# order children by startTime
sub["startTime"] = pd.to_datetime(sub["startTime"], errors="coerce", utc=True)
start_by_id = {r["id"]: r["startTime"] for _, r in sub.iterrows()}
for k in children:
    children[k].sort(key=lambda i: (start_by_id[i] is pd.NaT, start_by_id[i]))
roots.sort(key=lambda i: (start_by_id[i] is pd.NaT, start_by_id[i]))

# Flatten depth-first into ordered rows (cap depth/count for readability)
flat = []
def walk(nid, depth):
    if len(flat) >= 22:
        return
    r = by_id[nid]
    flat.append((depth, r))
    for c in children.get(nid, []):
        walk(c, depth + 1)

for rt in roots:
    walk(rt, 0)

# ---- Render Langfuse-style waterfall ----
TYPE_COLOR = {
    "SPAN": "#6366F1", "CHAIN": "#8B5CF6", "TOOL": "#0EA5E9",
    "GENERATION": "#10B981", "RETRIEVER": "#F59E0B",
    "EMBEDDING": "#EC4899", "AGENT": "#4F46E5",
}
max_lat = max((float(by_id[i]["latencyMs"]) for _, r in flat for i in [r["id"]]
               if pd.notna(by_id[i]["latencyMs"])), default=1.0) or 1.0

n = len(flat)
fig, ax = plt.subplots(figsize=(11, 0.46 * n + 0.8), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, n)
ax.axis("off")

row_h = 0.78
for idx, (depth, r) in enumerate(flat):
    y = n - idx - 1
    lat = float(r["latencyMs"]) if pd.notna(r["latencyMs"]) else 0.0
    typ = r["type"]
    color = TYPE_COLOR.get(typ, "#9CA3AF")
    indent = depth * 3.2
    # tree label
    name = str(r["name"])[:34]
    ax.text(indent + 0.4, y + 0.4, ("└ " if depth else "") + name,
            fontsize=9.5, va="center", ha="left",
            family="DejaVu Sans", color="#111827",
            fontweight="bold" if depth == 0 else "normal")
    # type pill
    ax.text(38.5, y + 0.4, typ, fontsize=7.2, va="center", ha="left",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"))
    # latency bar (waterfall)
    bar_w = max(0.6, (lat / max_lat) * 44.0)
    bar = FancyBboxPatch((50, y + 0.16), bar_w, row_h - 0.3,
                         boxstyle="round,pad=0.02", fc=color, ec="none", alpha=0.85)
    ax.add_patch(bar)
    # latency label
    lab = f"{lat/1000:.1f}s" if lat >= 1000 else f"{lat:.0f}ms"
    ax.text(50 + bar_w + 0.8, y + 0.4, lab, fontsize=8, va="center", ha="left",
            color="#374151")

# header
ax.text(0.4, n + 0.15, "Traza: generación de examen", fontsize=12,
        fontweight="bold", color="#1E1B4B", ha="left")
ax.text(50, n + 0.15, "latencia (waterfall)", fontsize=9, color="#6B7280", ha="left")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"✅ {OUT}")
print(f"   traceId: {best}")
print(f"   nodos renderizados: {n}")

# also dump a small summary for the slide caption
total_lat = sub[sub["id"].isin([r['id'] for _, r in flat])]
root_lat = float(by_id[roots[0]]["latencyMs"]) if pd.notna(by_id[roots[0]]["latencyMs"]) else None
meta = {
    "traceId": str(best),
    "n_observations_in_trace": int(len(sub)),
    "n_rendered": n,
    "root_name": str(by_id[roots[0]]["name"]),
    "root_latency_s": round(root_lat / 1000, 1) if root_lat else None,
}
json.dump(meta, open("/home/santiago/workspaces/ai-tutor/docs/observability/exam_trace_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
