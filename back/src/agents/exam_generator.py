"""ExamGenerator Agent — deterministic StateGraph for generating personalized exams.

Produces source-grounded exams (MCQ + open-answer) from ingested material.
Uses pure StateGraph topology (not ReAct): batch retrieval → single structured
LLM call → claim-level embedding validation → 3-retry loop → format.
"""

from __future__ import annotations

import operator
from datetime import UTC
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.config import settings

# ── Pydantic structured-output models ────────────────────────────────────────


class MCQQuestion(BaseModel):
    """Multiple-choice question with stem, options, and source grounding."""

    stem: str = Field(description="Question stem (the question text)")
    options: list[str] = Field(description="3-5 answer options, exactly one correct")
    correct_option_index: int = Field(description="0-based index of the correct option")
    source_chunk_ids: list[str] = Field(description="ChromaDB chunk IDs supporting this question")
    difficulty: str = Field(description="'easy' | 'medium' | 'hard'")
    topic: str = Field(description="Primary topic this question covers")


class OpenAnswerQuestion(BaseModel):
    """Open-ended question with base answer for evaluator grading."""

    prompt: str = Field(description="Open-ended question prompt")
    base_answer: str = Field(description="Expected answer — used by Evaluator (Epic 5) for grading")
    key_points: list[str] = Field(description="3-5 key points the answer should include")
    source_chunk_ids: list[str] = Field(description="ChromaDB chunk IDs supporting this question")
    difficulty: str = Field(description="'easy' | 'medium' | 'hard'")
    topic: str = Field(description="Primary topic this question covers")


class ExamGeneration(BaseModel):
    """Batch exam generation — all questions in a single structured LLM call."""

    mcq_questions: list[MCQQuestion] = Field(
        ..., description="MCQ questions list (required, can be empty)"
    )
    open_questions: list[OpenAnswerQuestion] = Field(
        ..., description="Open-answer questions list (required, can be empty)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="{topics_covered: [...], total_source_chunks: N}",
    )


# ── State schema ─────────────────────────────────────────────────────────────


class ExamGeneratorState(TypedDict):
    session_id: str
    student_id: str
    topics: list[str]
    difficulty: str
    question_count: int
    mcq_ratio: float
    student_profile: dict[str, Any] | None
    collection_name: str
    retrieved_chunks: Annotated[list[dict[str, Any]], operator.add]
    generated_questions: list[dict[str, Any]]
    validation_results: list[dict[str, Any]]
    validation_errors: Annotated[list[str], operator.add]
    invalid_question_indices: list[int]
    omitted_questions: Annotated[list[int], operator.add]
    retry_count: int
    topic_not_found: list[str]
    topic_suggestions: list[str]
    topic_distribution: dict[str, int]
    exam: dict[str, Any]
    status: str


# ── Helper for theme matching ────────────────────────────────────────────────


def _deep_merge_tree(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Deep-merge *source* nested dict into *target*, mutating *target*."""
    for key, value in source.items():
        if key not in target:
            target[key] = {}
        if isinstance(value, dict) and isinstance(target[key], dict):
            _deep_merge_tree(target[key], value)


def match_theme_to_session_topics(theme: str, session_topics: list[str]) -> list[str]:
    """Match a user-requested theme to one or many session topics.

    Returns matching topics if any are found, otherwise returns an empty list.
    """
    theme_clean = theme.strip().lower()
    if not theme_clean or not session_topics:
        return []

    # 1. Exact match (case-insensitive)
    exact = [t for t in session_topics if t.strip().lower() == theme_clean]
    if exact:
        return exact

    # 2. Segment/path match (e.g. if theme is "derivadas" and session topic is "cálculo/derivadas")
    path_matches = []
    for t in session_topics:
        parts = [p.strip().lower() for p in t.split("/")]
        if theme_clean in parts:
            path_matches.append(t)
    if path_matches:
        return path_matches

    # 3. Substring match
    substring_matches = []
    for t in session_topics:
        t_clean = t.strip().lower()
        if theme_clean in t_clean or t_clean in theme_clean:
            substring_matches.append(t)
    if substring_matches:
        return substring_matches

    # 4. Jaccard similarity match on stems
    try:
        from src.topic_extraction.preprocess import jaccard_similarity, stem_topic

        theme_stem = stem_topic(theme)
        jaccard_matches = []
        for t in session_topics:
            topic_stem = stem_topic(t)
            sim = jaccard_similarity(theme_stem, topic_stem)
            if sim >= 0.4:
                jaccard_matches.append((t, sim))
        if jaccard_matches:
            # Sort by similarity score descending
            jaccard_matches.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in jaccard_matches]
    except Exception:
        pass

    return []


# ── Node implementations ─────────────────────────────────────────────────────


async def retrieve_relevant_chunks(state: ExamGeneratorState) -> dict[str, Any]:
    """Retrieve top-K relevant chunks from ChromaDB for the requested topics.

    Iterates state["topics"], calling retrieve_chunks per topic. Handles empty
    results by querying ThematicIndex for close-topic suggestions. Deduplicates
    by chunk_id. Weak topics from student_profile get 2× retrieval weight.
    """
    import logging

    from src.tools import retrieve_chunks as _retrieve_chunks

    logger = logging.getLogger(__name__)

    try:
        topics: list[str] = state.get("topics", [])
        session_id: str = state.get("session_id", "")
        collection_name = state.get("collection_name") or f"session_{session_id}"
        student_profile = state.get("student_profile")

        # Match user-requested theme/topics to topics extracted from session files
        topic_descriptions: dict[str, str] | None = None
        topic_tree: dict[str, Any] | None = None
        if session_id and topics:
            from src.memory.schema import list_session_files
            from src.utils.async_ import run_async_in_sync
            import json as _json

            try:
                session_files = run_async_in_sync(list_session_files(session_id))
                session_topics = []
                for sf in session_files:
                    topics_json = sf.get("topics_json")
                    if topics_json:
                        try:
                            t_list = _json.loads(topics_json)
                            if isinstance(t_list, list):
                                session_topics.extend(t_list)
                        except Exception:
                            pass
                session_topics = list(set(session_topics))

                # Build topic_descriptions and topic_tree from session files
                _all_descs: dict[str, str] = {}
                _merged_tree: dict[str, Any] = {}
                for sf in session_files:
                    descs_json = sf.get("topic_descriptions_json")
                    if descs_json:
                        try:
                            sf_descs = _json.loads(descs_json)
                            if isinstance(sf_descs, dict):
                                for k, v in sf_descs.items():
                                    if isinstance(v, str) and v.strip():
                                        _all_descs[k] = v
                        except Exception:
                            pass
                    tree_json = sf.get("topic_tree_json")
                    if tree_json:
                        try:
                            sf_tree = _json.loads(tree_json)
                            if isinstance(sf_tree, dict) and sf_tree:
                                _deep_merge_tree(_merged_tree, sf_tree)
                        except Exception:
                            pass
                if _all_descs:
                    topic_descriptions = _all_descs
                if _merged_tree:
                    topic_tree = _merged_tree

                if session_topics:
                    resolved_topics: list[str] = []
                    has_any_match = False

                    # Try LLM-based fuzzy matching when descriptions available
                    if topic_descriptions:
                        try:
                            from src.rag import match_user_topics_to_session

                            llm_matches = await match_user_topics_to_session(
                                topics, session_topics, topic_descriptions
                            )
                            if llm_matches:
                                resolved_topics = list(llm_matches.keys())
                                has_any_match = True
                                logger.info(
                                    "LLM matched themes %s → session topics %s",
                                    topics, resolved_topics,
                                )
                        except Exception:
                            logger.warning("LLM topic matching failed, falling back to token matching")

                    # Fall back to token-based matching if LLM didn't match
                    if not has_any_match:
                        for theme in topics:
                            matches = match_theme_to_session_topics(theme, session_topics)
                            if matches:
                                resolved_topics.extend(matches)
                                has_any_match = True
                            else:
                                resolved_topics.append(theme)

                    if has_any_match:
                        # Deduplicate preserving order
                        seen_resolved = set()
                        deduped_resolved = []
                        for rt in resolved_topics:
                            if rt not in seen_resolved:
                                seen_resolved.add(rt)
                                deduped_resolved.append(rt)
                        logger.info(
                            "Matched themes %s to session topics %s", topics, deduped_resolved
                        )
                        topics = deduped_resolved
            except Exception as e:
                logger.warning("Failed to resolve theme to session topics: %s", e)

        # Determine topics with optional weak-topic boosting
        weak_topics: list[str] = []
        if student_profile and isinstance(student_profile, dict):
            weak_topics = student_profile.get("weak_topics", [])

        # Build weighted topic list: weak topics appear twice
        weighted_topics: list[str] = list(topics)
        for wt in weak_topics:
            if wt not in topics:
                weighted_topics.append(wt)
            weighted_topics.append(wt)  # 2× weight

        # If no topics at all, use weak topics
        if not weighted_topics and weak_topics:
            weighted_topics = list(set(weak_topics))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_topics: list[str] = []
        for t in weighted_topics:
            if t not in seen:
                seen.add(t)
                unique_topics.append(t)

        # Retrieve chunks for each topic
        all_chunks: list[dict[str, Any]] = []
        topic_not_found: list[str] = []
        seen_chunk_ids: set[str] = set()

        for topic in unique_topics:
            chunks = _retrieve_chunks.invoke(
                {
                    "query": topic,
                    "top_k": settings.retrieval_top_k,
                    "collection_name": collection_name,
                    "topic_descriptions": topic_descriptions,
                    "topic_tree": topic_tree if topic_tree else None,
                }
            )
            if not chunks:
                topic_not_found.append(topic)
            else:
                # Check if retrieved chunks are semantically relevant to the topic.
                # similarity_score is cosine distance (lower = more similar).
                # If the best match has distance > 0.70, the topic is effectively absent.
                best_distance = min((c.get("similarity_score", 1.0) for c in chunks), default=1.0)
                if best_distance > 0.60:
                    topic_not_found.append(topic)
                    continue
                for chunk in chunks:
                    cid = chunk.get("chunk_id", "")
                    if cid and cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        all_chunks.append(chunk)

        # For missing topics, query ThematicIndex for suggestions
        topic_suggestions: list[str] = []
        if topic_not_found:
            try:
                from src.rag import ThematicIndex as _ThematicIndex

                # Build a temporary index from chunk metadata
                ti = _ThematicIndex()
                for chunk in all_chunks:
                    meta_topic = (
                        chunk.get("metadata", {}).get("topic", "")
                        if isinstance(chunk.get("metadata"), dict)
                        else ""
                    )
                    if meta_topic:
                        ti.add_topics([meta_topic])

                for missing in topic_not_found:
                    # Try searching with each segment of the missing topic path
                    parts = missing.split("/")
                    for i in range(len(parts)):
                        prefix = "/".join(parts[: i + 1])
                        if i == 0:
                            suggestions = list(ti.to_dict().keys())[:3]
                        else:
                            suggestions = ti.search(prefix)
                        if suggestions:
                            topic_suggestions.extend(suggestions[:3])
                            break

                topic_suggestions = list(dict.fromkeys(topic_suggestions))[:3]
            except Exception:
                logger.debug("ThematicIndex suggestion lookup failed", exc_info=True)

        return {
            "topics": topics,
            "retrieved_chunks": all_chunks,
            "topic_not_found": topic_not_found,
            "topic_suggestions": topic_suggestions,
            "collection_name": collection_name,
            "status": "retrieved" if all_chunks else "no_material",
        }

    except Exception as exc:
        logger.exception("retrieve_relevant_chunks failed")
        return {
            "validation_errors": [f"Retrieval error: {exc}"],
            "status": "error",
        }


def generate_questions(
    state: ExamGeneratorState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Generate exam questions via a single structured LLM call.

    Builds a prompt from retrieved chunks, user preferences, student profile,
    and requested topics/difficulty. Uses ChatGroq with structured_output to
    get a deterministic ExamGeneration. On retry, only invalid slots are
    regenerated — valid questions from previous passes are preserved.
    """
    import logging
    import time

    from src.llm import get_structured_llm
    from src.rag.policy import RAG_ONLY_SYSTEM_PROMPT

    logger = logging.getLogger(__name__)
    t0 = time.monotonic()

    try:
        chunks: list[dict[str, Any]] = state.get("retrieved_chunks", [])
        # Guard: no chunks → cannot generate
        if not chunks:
            return {
                "generated_questions": [],
                "topic_distribution": {},
                "status": "no_material",
            }

        question_count: int = state.get("question_count", 5)
        mcq_ratio: float = state.get("mcq_ratio", 0.5)
        difficulty: str = state.get("difficulty", "medium")

        logger.info(
            "[generate_questions] START | session=%s | topic=%s | count=%d",
            state["session_id"],
            state.get("topic", "?"),
            question_count,
        )

        student_profile = state.get("student_profile")
        retry_count: int = state.get("retry_count", 0)
        invalid_indices: list[int] = state.get("invalid_question_indices", [])
        existing_questions: list[dict[str, Any]] = state.get("generated_questions", [])
        validation_errors: list[str] = state.get("validation_errors", [])

        # On retry: only generate replacements for invalid slots
        target_count = (
            len(invalid_indices) if retry_count > 0 and invalid_indices else question_count
        )

        if target_count <= 0:
            return {"generated_questions": existing_questions, "status": "generated"}

        # Build chunk context
        chunk_context = "\n\n".join(
            f"[CHUNK:{c.get('chunk_id', '?')}] {c.get('text', '')}" for c in chunks
        )[:8000]  # Truncate to avoid token overflow

        # Build preferences section
        prefs_lines: list[str] = []
        if mcq_ratio <= 0.0:
            mcq_n = 0
        elif mcq_ratio >= 1.0:
            mcq_n = target_count
        else:
            mcq_n = round(target_count * mcq_ratio)
            if mcq_n == 0 and target_count > 0:
                mcq_n = 1
            if mcq_n == target_count and target_count > 1:
                mcq_n = target_count - 1
        open_n = target_count - mcq_n
        prefs_lines.append(f"Total questions: {target_count}")
        prefs_lines.append(f"MCQ questions: {mcq_n}")
        prefs_lines.append(f"Open-answer questions: {open_n}")
        prefs_lines.append(f"Difficulty: {difficulty}")

        if student_profile and isinstance(student_profile, dict):
            weak = student_profile.get("weak_topics", [])
            if weak:
                prefs_lines.append(f"Prioritize weak topics: {', '.join(weak)}")

        # ── Compute topic distribution: weak-topic bias or uniform ──
        available_topics: set[str] = set()
        for c in chunks:
            meta = c.get("metadata", {})
            if isinstance(meta, dict) and meta.get("topic"):
                available_topics.add(meta["topic"])
        if not available_topics:
            available_topics = set(state.get("topics", []))

        weak_topics_set: set[str] = set()
        if student_profile and isinstance(student_profile, dict):
            weak_topics_set = {
                wt for wt in student_profile.get("weak_topics", []) if wt in available_topics
            }

        topic_distribution: dict[str, int] = {}
        if weak_topics_set and available_topics:
            # ≥60% to weak topics
            weak_list = sorted(weak_topics_set)
            strong_list = sorted(available_topics - weak_topics_set)
            weak_quota = max(1, round(target_count * 0.6))
            weak_quota = min(weak_quota, target_count)

            # Distribute weak quota evenly across weak topics
            per_weak = weak_quota // len(weak_list)
            weak_rem = weak_quota % len(weak_list)
            for i, wt in enumerate(weak_list):
                topic_distribution[wt] = per_weak + (1 if i < weak_rem else 0)

            # Distribute remaining across strong topics
            strong_quota = target_count - weak_quota
            if strong_list and strong_quota > 0:
                per_strong = strong_quota // len(strong_list)
                strong_rem = strong_quota % len(strong_list)
                for i, st in enumerate(strong_list):
                    topic_distribution[st] = per_strong + (1 if i < strong_rem else 0)
            elif not strong_list:
                # All available topics are weak — spread remaining evenly
                extra_per = strong_quota // len(weak_list)
                extra_rem = strong_quota % len(weak_list)
                for i, wt in enumerate(weak_list):
                    topic_distribution[wt] += extra_per + (1 if i < extra_rem else 0)
        elif available_topics:
            # Uniform distribution across all available topics
            topic_list = sorted(available_topics)
            per_topic = target_count // len(topic_list)
            rem = target_count % len(topic_list)
            for i, t in enumerate(topic_list):
                topic_distribution[t] = per_topic + (1 if i < rem else 0)

        if topic_distribution:
            dist_lines = ["Topic distribution:"]
            for t, count in topic_distribution.items():
                dist_lines.append(f"  - {t}: {count} question(s)")
            prefs_lines.extend(dist_lines)

        # Retry instructions
        retry_instructions = ""
        if retry_count > 0 and invalid_indices:
            retry_instructions = (
                f"\nIMPORTANTE: Solo generá preguntas para los índices {invalid_indices}. "
                f"Las preguntas anteriores fallaron validación porque sus afirmaciones "
                f"no coincidían con los chunks fuente. Asegurate de que CADA afirmación "
                f"provenga directamente de un chunk etiquetado con [CHUNK:...].\n"
                f"Errores anteriores: {'; '.join(validation_errors[-5:])}"
            )

        header = RAG_ONLY_SYSTEM_PROMPT
        req_open = (
            "- Para open-answer: prompts que requieran explicación "
            "(no sí/no), incluir base_answer y 3-5 key_points."
        )
        prompt = f"""{header}

{chunk_context}

PREFERENCIAS:
{chr(10).join(prefs_lines)}{retry_instructions}

REQUISITOS:
- Cada pregunta DEBE basarse en hechos textuales de los chunks provistos.
- Para MCQs: 3-5 opciones, exactamente una correcta, distractores plausibles.
{req_open}
- Incluí source_chunk_ids (los IDs entre corchetes [CHUNK:xxx]) por cada pregunta.
- Cada pregunta debe tener los campos: topic y difficulty.
"""

        structured_llm = get_structured_llm(
            ExamGeneration, temperature=settings.exam_generator_temperature
        )
        invoke_kwargs = {"config": config} if config is not None else {}
        result: ExamGeneration = structured_llm.invoke(prompt, **invoke_kwargs)

        # Convert Pydantic models to dicts
        new_questions: list[dict[str, Any]] = []
        for mcq in result.mcq_questions:
            q = mcq.model_dump()
            q["type"] = "mcq"
            q["prompt"] = q.pop("stem")  # map internal field to API field
            new_questions.append(q)
        for oa in result.open_questions:
            q = oa.model_dump()
            q["type"] = "open_answer"
            new_questions.append(q)

        # ── Post-filter: remove hallucinated source_chunk_ids ──────────
        valid_chunk_ids = {c.get("chunk_id", "") for c in chunks}

        for q in new_questions:
            raw_ids = q.get("source_chunk_ids", [])
            if not raw_ids:
                continue
            filtered = [cid for cid in raw_ids if cid in valid_chunk_ids]
            removed = set(raw_ids) - set(filtered)
            if removed:
                logger.warning(
                    "[generate_questions] Hallucinated chunk ID(s) removed: %s | question=%s",
                    sorted(removed),
                    q.get("prompt", q.get("stem", ""))[:80],
                )
            q["source_chunk_ids"] = filtered

        # On retry: merge new questions into existing, replacing invalid slots
        if retry_count > 0 and invalid_indices and existing_questions:
            merged = list(existing_questions)
            for idx, new_q in zip(invalid_indices, new_questions):
                if idx < len(merged):
                    merged[idx] = new_q
                else:
                    merged.append(new_q)
            final_questions = merged
        else:
            final_questions = new_questions

        # Increment retry_count on every call (starts at 0, increments per retry)
        next_retry = retry_count + 1 if retry_count > 0 else 1

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "[generate_questions] COMPLETE | session=%s | questions=%d | %dms",
            state["session_id"],
            len(final_questions),
            int(elapsed),
        )
        return {
            "generated_questions": final_questions,
            "retry_count": next_retry,
            "topic_distribution": topic_distribution,
            "status": "generated",
        }

    except Exception as exc:
        logger.exception("generate_questions failed")
        return {
            "validation_errors": [f"Generation error: {exc}"],
            "status": "error",
        }


def validate_questions(state: ExamGeneratorState) -> dict[str, Any]:
    """Validate every question by checking claims against source chunks.

    Extracts atomic claims from each question (stem + options for MCQ,
    base_answer + key_points for open-answer). Delegates to
    ``validate_claim_grounding`` (retry_trigger mode) for batch
    embedding and cosine similarity via ``cos_sim``.
    Claims below the anti_hallucination_threshold are flagged as errors.
    """
    import logging

    from src.tools.validate_claim_grounding import validate_claim_grounding
    from src.utils.text import split_into_claims

    logger = logging.getLogger(__name__)

    try:
        chunks: list[dict[str, Any]] = state.get("retrieved_chunks", [])
        questions: list[dict[str, Any]] = state.get("generated_questions", [])

        if not questions:
            return {
                "validation_results": [],
                "validation_errors": [],
                "invalid_question_indices": [],
            }

        # ── Extract claims per question, tracking ownership ──
        all_claims: list[str] = []
        claim_to_question: list[int] = []

        for qi, question in enumerate(questions):
            qtype = question.get("type", "")
            q_claims: list[str] = []

            if qtype == "mcq":
                stem = question.get("stem", "")
                q_claims.extend(split_into_claims(stem, min_length=20))
                for opt in question.get("options", []):
                    opt_clean = opt.strip()
                    if len(opt_clean) >= 20:
                        q_claims.append(opt_clean)
            else:  # open_answer
                base = question.get("base_answer", "")
                q_claims.extend(split_into_claims(base, min_length=20))
                for kp in question.get("key_points", []):
                    kp_clean = kp.strip()
                    if len(kp_clean) >= 10:
                        q_claims.append(kp_clean)

            # Fallback: if no claims, use full stem/base_answer
            if not q_claims:
                if qtype == "mcq":
                    fallback = [question.get("stem", "")]
                else:
                    fallback = [question.get("base_answer", "")]
                q_claims = [c for c in fallback if len(c.strip()) >= 10]

            for claim in q_claims:
                all_claims.append(claim)
                claim_to_question.append(qi)

        # ── All empty: pass validation ──
        if not all_claims:
            return {
                "validation_results": [
                    {
                        "question_index": qi,
                        "valid": True,
                        "claims_checked": [],
                        "missing_claims": [],
                        "matched_chunk_ids": [],
                    }
                    for qi in range(len(questions))
                ],
                "validation_errors": [],
                "invalid_question_indices": [],
            }

        # ── Delegate to anti-hallucination tool ──
        result = validate_claim_grounding.invoke(
            {
                "claims": all_claims,
                "chunks": chunks,
                "mode": "retry_trigger",
            }
        )

        # ── Organize tool results per question ──
        claim_results = result.get("claim_results", [])
        per_question: dict[int, dict[str, Any]] = {}
        for qi in range(len(questions)):
            per_question[qi] = {
                "claim_checks": [],
                "missing_claims": [],
                "matched_chunk_ids": [],
                "all_matched": True,
            }

        for ci, cr in enumerate(claim_results):
            qi = claim_to_question[ci] if ci < len(claim_to_question) else 0
            claim_text = cr["claim_text"]
            matched = cr["matched"]
            best_chunk_id = cr["best_chunk_id"]

            pq = per_question[qi]
            if not matched:
                pq["all_matched"] = False
                pq["missing_claims"].append(claim_text[:80])

            if best_chunk_id and best_chunk_id not in pq["matched_chunk_ids"]:
                pq["matched_chunk_ids"].append(best_chunk_id)

            pq["claim_checks"].append(
                {
                    "claim_text": claim_text,
                    "best_chunk_id": best_chunk_id,
                    "similarity_score": cr["similarity_score"],
                    "matched": matched,
                }
            )

        # ── Build final output ──
        validation_results: list[dict[str, Any]] = []
        all_errors: list[str] = []
        invalid_indices: list[int] = []

        for qi in range(len(questions)):
            pq = per_question[qi]
            validation_results.append(
                {
                    "question_index": qi,
                    "valid": pq["all_matched"],
                    "claims_checked": pq["claim_checks"],
                    "missing_claims": pq["missing_claims"],
                    "matched_chunk_ids": pq["matched_chunk_ids"],
                }
            )
            if not pq["all_matched"]:
                invalid_indices.append(qi)
                all_errors.append(
                    f"Question {qi}: {len(pq['missing_claims'])} "
                    "claim(s) not found in source chunks"
                )

        return {
            "validation_results": validation_results,
            "validation_errors": all_errors,
            "invalid_question_indices": invalid_indices,
        }

    except Exception as exc:
        logger.exception("validate_questions failed")
        return {
            "validation_errors": [f"Validation error: {exc}"],
        }


def should_retry(state: ExamGeneratorState) -> str:
    """Return 'retry' if validation errors exist AND retry_count < 3, else 'done'.

    Delegates to ``src.utils.retry.should_retry`` — the single source of truth
    for retry-decision logic across all agents.
    """
    from src.utils.retry import should_retry as _should_retry

    return _should_retry(
        validation_errors=state.get("validation_errors", []),
        retry_count=state.get("retry_count", 0),
        status=state.get("status", ""),
    )


def format_exam(state: ExamGeneratorState) -> dict[str, Any]:
    """Package validated questions into final exam dict with metadata.

    Removes omitted questions, computes topic coverage, adds warnings,
    and sets status (complete | partial | no_material).
    """
    import uuid as _uuid
    from datetime import datetime

    try:
        questions: list[dict[str, Any]] = state.get("generated_questions", [])
        invalid_indices: list[int] = state.get("invalid_question_indices", [])
        omitted_indices: list[int] = list(state.get("omitted_questions", []))
        validation_errors: list[str] = state.get("validation_errors", [])
        topic_not_found: list[str] = state.get("topic_not_found", [])
        topic_suggestions: list[str] = state.get("topic_suggestions", [])
        retry_count: int = state.get("retry_count", 0)

        # If retries exhausted and questions still fail, mark for omission
        if retry_count >= 3 and invalid_indices:
            for idx in invalid_indices:
                if idx not in omitted_indices:
                    omitted_indices.append(idx)

        # Filter out omitted questions
        omitted_set = set(omitted_indices)
        final_questions = [q for i, q in enumerate(questions) if i not in omitted_set]

        # Compute topics covered
        topics_covered = list(
            dict.fromkeys(q.get("topic", "") for q in final_questions if q.get("topic"))
        )

        # Compute actual topic distribution from final questions
        topic_distribution: dict[str, int] = {}
        for q in final_questions:
            qtopic = q.get("topic", "")
            if qtopic:
                topic_distribution[qtopic] = topic_distribution.get(qtopic, 0) + 1

        # Determine status
        if not final_questions:
            exam_status = "no_material"
        elif topic_not_found and not topics_covered:
            exam_status = "no_material"
        elif omitted_indices or validation_errors:
            exam_status = "partial"
        else:
            exam_status = "complete"

        exam = {
            "exam_id": str(_uuid.uuid4()),
            "session_id": state.get("session_id", ""),
            "student_id": state.get("student_id", ""),
            "generated_at": datetime.now(UTC).isoformat(),
            "total_questions": len(final_questions),
            "questions": final_questions,
            "topics_covered": topics_covered,
            "topic_distribution": topic_distribution,
            "source_chunks_total": len(state.get("retrieved_chunks", [])),
            "omitted_count": len(omitted_indices),
            "topic_not_found": topic_not_found,
            "topic_suggestions": topic_suggestions,
            "status": exam_status,
            "warnings": validation_errors,
        }

        return {"exam": exam, "status": exam_status}
    except Exception as exc:
        import logging

        logger = logging.getLogger(__name__)
        logger.exception("format_exam failed")
        return {
            "exam": {"error": str(exc), "status": "error"},
            "status": "error",
            "validation_errors": [f"Format error: {exc}"],
        }


def build_exam_generator() -> StateGraph[ExamGeneratorState, Any, Any]:
    """Build and return the ExamGenerator LangGraph."""
    builder = StateGraph(ExamGeneratorState)

    builder.add_node("retrieve_relevant_chunks", retrieve_relevant_chunks)
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("validate_questions", validate_questions)
    builder.add_node("format_exam", format_exam)

    builder.add_edge(START, "retrieve_relevant_chunks")
    builder.add_edge("retrieve_relevant_chunks", "generate_questions")
    builder.add_edge("generate_questions", "validate_questions")
    builder.add_conditional_edges(
        "validate_questions",
        should_retry,
        {"retry": "generate_questions", "done": "format_exam"},
    )
    builder.add_edge("format_exam", END)

    return builder
