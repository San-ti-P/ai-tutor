---
name: framework-selection
description: "INVOKE THIS SKILL at the START of any LangChain/LangGraph project, before writing any agent code. Determines whether to use LangChain directly, LangGraph, or a combination. Must be consulted before other agent skills."
---

<overview>
LangGraph builds on LangChain — they are **layered**, not competing. Pick the right tool for each part of the system:

```
┌─────────────────────────────────────────┐
│               LangGraph                 │  ← orchestration: graphs, loops, state
│    (nodes, edges, state, persistence)   │
├─────────────────────────────────────────┤
│               LangChain                 │  ← foundation: models, tools, chains
│      (models, tools, prompts, RAG)      │
└─────────────────────────────────────────┘
```

Picking LangGraph does not cut you off from LangChain — use LangChain primitives (tools, retrievers, chains) freely inside LangGraph nodes.

> **This skill should be loaded at the top of any project before selecting other skills or writing agent code.** The framework you choose dictates which other skills to invoke next.
</overview>

---

## Decision Guide

<decision-table>

Answer these questions in order:

| Question | Yes → | No → |
|----------|-------|-------|
| Does the task require complex control flow — loops, dynamic branching, parallel workers, human-in-the-loop, or custom state? | **LangGraph** | ↓ |
| Is this a single-purpose agent that takes input, runs tools, and returns a result? | **LangChain** (`create_agent`) | ↓ |
| Is this a pure model call, chain, or retrieval pipeline with no agent loop? | **LangChain** (chain) | — |

</decision-table>

---

## Framework Profiles

### LangChain — Use when the task is focused and self-contained

**Best for:**
- Single-purpose agents that use a fixed set of tools
- RAG pipelines and document Q&A
- Model calls, prompt templates, output parsing
- Quick prototypes where agent logic is simple

**Not ideal when:**
- The agent needs to plan across many steps
- State needs to persist across multiple sessions
- Control flow is conditional or iterative

**Skills to invoke next:** `langchain-rag`, `langchain-middleware`

### LangGraph — Use when you need to own the control flow

**Best for:**
- Agents with branching logic or loops (e.g. retry-until-correct, reflection)
- Multi-step workflows where different paths depend on intermediate results
- Human-in-the-loop approval at specific steps
- Parallel fan-out / fan-in (map-reduce patterns)
- Persistent state across invocations within a session

**Not ideal when:**
- The workflow is straightforward enough for a simple agent

**Skills to invoke next:** `langgraph-fundamentals`, `langgraph-human-in-the-loop`, `langgraph-persistence`

---

## Mixing Layers

<mixing-layers>
LangChain and LangGraph work together in the same project. The standard pattern for multi-agent systems:

### When to mix

| Scenario | Recommended pattern |
|----------|---------------------|
| Simple RAG retrieval or tool call inside a larger orchestrated flow | LangGraph node uses LangChain retriever/chain |
| Multi-agent system with orchestrator + specialized agents | LangGraph StateGraph with nodes per agent role |
| Single step doesn't need graph overhead | Use `create_agent()` or chain directly inside a LangGraph node |

### How it works in practice

LangGraph nodes can contain LangChain chains, retrievers, and tool-calling agents. The node's return dict merges into the shared StateGraph state. Example:

```python
# LangGraph node wrapping a LangChain retriever
def retrieve_node(state: State) -> dict:
    retriever = chroma_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(state["query"])
    return {"context": docs}
```

LangChain tools are shared building blocks — define them once, use them in any LangGraph node. Tool definitions are agent-agnostic.
</mixing-layers>

---

## Quick Reference

<quick-reference>

| | LangChain | LangGraph |
|---|-----------|-----------|
| **Control flow** | Fixed (tool loop) | Custom (graph) |
| **Middleware** | Callbacks only | Wire behavior into nodes/edges |
| **Planning** | ✗ | Manual (build plan as graph) |
| **Persistent memory** | ✗ | With checkpointer |
| **Subagent delegation** | ✗ | Manual (Send API, subgraphs) |
| **Human-in-the-loop** | ✗ | Manual interrupt |
| **Custom graph edges** | ✗ | ✓ Full control |
| **Setup complexity** | Low | Medium |
| **Flexibility** | Medium | High |

</quick-reference>

---

## Project-Specific Guidance (Tutor Académico Personal)

This project uses **LangGraph as the primary orchestration layer** with LangChain primitives for RAG, tool definitions, and model calls. The 6-agent architecture maps cleanly:

| Agent | LangGraph Pattern | LangChain Usage |
|-------|-------------------|----------------|
| Orchestrator | Plan-and-Execute graph | `create_agent()` for intent classification |
| Ingestor | ReAct loop node | Chains for document parsing, embeddings |
| ExamGenerator | ReAct + tool-calling node | RAG retrievers, structured output |
| ExerciseGenerator | ReAct + tool-calling node | RAG retrievers |
| Evaluator | Chain-of-Thought node | Prompt chains, structured scoring |
| Support Agent | Reactive node | SQLite via tools |

**Rule**: Use LangGraph for any multi-step agent flow. Use LangChain directly only for simple, single-step operations (embedding, retrieval, single model call).
