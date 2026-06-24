#!/usr/bin/env python3
"""Render the REAL LangGraph graph of each agent to PNG (from the dev code)."""
import os
import sys

sys.path.insert(0, "/home/santiago/workspaces/ai-tutor/back")
os.chdir("/home/santiago/workspaces/ai-tutor/back")

OUT = "/home/santiago/workspaces/ai-tutor/docs/presentation/assets"
os.makedirs(OUT, exist_ok=True)

from src.agents.orchestrator import build_orchestrator
from src.agents.ingestor import build_ingestor
from src.agents.exam_generator import build_exam_generator
from src.agents.exercise_generator import build_exercise_generator
from src.agents.evaluator import build_evaluator
from src.agents.support import build_support_agent

builders = {
    "orchestrator": build_orchestrator,
    "ingestor": build_ingestor,
    "exam_generator": build_exam_generator,
    "exercise_generator": build_exercise_generator,
    "evaluator": build_evaluator,
    "support": build_support_agent,
}

for name, build in builders.items():
    try:
        graph = build().compile()
        g = graph.get_graph()
        # Save mermaid text always
        mmd = g.draw_mermaid()
        with open(f"{OUT}/{name}.mmd", "w") as f:
            f.write(mmd)
        # Try PNG via mermaid.ink (needs network)
        try:
            png = g.draw_mermaid_png()
            with open(f"{OUT}/graph_{name}.png", "wb") as f:
                f.write(png)
            print(f"✅ {name}: PNG + mermaid")
        except Exception as e:
            print(f"⚠️  {name}: mermaid OK, PNG failed ({type(e).__name__})")
    except Exception as e:
        print(f"❌ {name}: build failed — {e}")
