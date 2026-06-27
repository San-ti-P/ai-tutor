# Manual Test Checklist — AI Tutor

Step-by-step validation procedure for human or LLM agents. Covers all agent workflows end-to-end.

**Prerequisites**: Backend running (`uv run uvicorn src.main:app --port 8000`), frontend running (`npm run dev`), LLM provider configured (Ollama or Groq), at least one PDF in `back/tests/fixtures/apunteAgentes_IA2007.pdf`.

---

## Quick Smoke Test (5 items, <3 minutes)

Run this before any demo or defense presentation.

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| S1 | Open `http://localhost:3000` | App loads, chat input visible, sidebar shows | Page renders without errors in browser console |
| S2 | Click "Nueva sesión", name it "Smoke Test", create | Session appears in sidebar, highlighted as active | Sidebar shows "Smoke Test" with accent color |
| S3 | Upload `apunteAgentes_IA2007.pdf` via paperclip button | Upload progress indicator, confirmation with topic count | Toast/notification shows "X temas detectados" |
| S4 | Type "dame un examen de 3 preguntas" and send | Exam renders with 3 questions (MCQ or free-text) | 3 question cards visible with answer inputs |
| S5 | Check Dashboard tab | Page loads, shows session count ≥ 1 | No spinner forever, no crash, stats cards render |

If all S1-S5 pass → system is demo-ready. If any fail → run full checklist for that section.

---

## Full Validation Checklist

### 1. Backend Health

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| 1.1 | `curl http://localhost:8000/health` | `{"status": "ok"}` | HTTP 200, JSON body |
| 1.2 | `curl http://localhost:8000/docs` | OpenAPI docs load | Browser shows interactive Swagger UI |
| 1.3 | Check backend logs on startup | `START orchestrator node: load_profile` logged | Terminal shows structured log lines with timestamps |
| 1.4 | Verify LLM provider reachable | No crash on first LLM call | Backend log shows "LLM call" span without connection errors |

### 2. Session Lifecycle

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| 2.1 | Open app, observe sidebar | Shows existing sessions or empty state | Sidebar visible on left, "Sesiones" header |
| 2.2 | Click "Nueva sesión" | Modal opens with name (required) + description fields | Input fields visible, create button disabled until name filled |
| 2.3 | Type name "IA Parcial 1", create | Session appears in sidebar, highlighted as active | Sidebar shows "IA Parcial 1" with accent border/color |
| 2.4 | Create second session "IA Parcial 2" | Two sessions in sidebar, "IA Parcial 2" active | Clicking between them switches active indicator |
| 2.5 | Rename session via pencil icon | Inline edit activates, Enter saves | New name persists after page reload (check localStorage) |
| 2.6 | Delete "IA Parcial 2" via trash icon | Confirmation prompt, session removed from sidebar | Sidebar no longer shows deleted session |
| 2.7 | Refresh page | Active session restored from localStorage | Same session highlighted, chat context preserved |

### 3. Document Ingestion

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| 3.1 | Click paperclip in active session | File picker opens | Native OS file dialog |
| 3.2 | Select `apunteAgentes_IA2007.pdf` | Upload progress visible | Progress bar or spinner during upload |
| 3.3 | Wait for confirmation | Toast: "N temas detectados, C chunks generados" | `N ≥ 3`, `C ≥ 10` |
| 3.4 | Check file list under session | PDF filename appears with classification badge | Badge shows "apunte teórico" or similar |
| 3.5 | Check topic tree renders | Hierarchical topic list visible (collapsible tree or flat badges) | At least 3 topic entries visible, not just empty container |
| 3.6 | Upload second file `apunteAgentes_IA2007.pdf` again | Incremental ingestion: topics merged, no duplicates | Topic count increases but doesn't double (merging) |
| 3.7 | Check browser Network tab | `/api/ingest` returns 200 with topics + topic_tree | Response JSON has `topics` array and `topic_tree` object |

### 4. Chat + Intent Classification

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| 4.1 | Type "hola" in chat | Agent responds with greeting in Spanish | Response text in Spanish, friendly tone |
| 4.2 | Type "qué temas tenés cargados" | Agent lists topics from ingested material | Topic names match those from ingest confirmation |
| 4.3 | Type "quiero estudiar agentes inteligentes" | Agent acknowledges and offers to generate exam/exercise | Response references "examen" or "ejercicio" or "preguntas" |
| 4.4 | Check short-term memory: type "qué te pregunté antes" | Agent references the previous message about agentes inteligentes | Response mentions prior topic or question |
| 4.5 | Check Langfuse trace (if enabled) | Trace for this session shows all LLM calls + tools | Langfuse dashboard shows spans for classify_intent, route, synthesize |

### 5. Exam Generation

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| 5.1 | Type "generame un examen de 5 preguntas sobre agentes inteligentes" | Exam widget renders with 5 questions | 5 question cards, each with question text + answer input |
| 5.2 | Verify question types | Mix of MCQ (radio buttons) and free-text (textarea) | At least 2 different input types visible |
| 5.3 | Verify questions reference material | Questions use concepts from PDF (agentes, taxonomía, entorno, etc.) | Spot-check: at least 2 questions contain terms from the uploaded PDF |
| 5.4 | Check no hallucination guardrail | No questions about topics not in the PDF (e.g., "redes neuronales") | All question topics appear in the topic tree from step 3.5 |
| 5.5 | Request exam on non-existent topic "física cuántica" | Agent responds: "No tengo material sobre física cuántica" + suggests alternatives | Response mentions missing topic and offers topics from actual material |
| 5.6 | Generate exam with specific count "3 preguntas" | Exactly 3 questions rendered | `questions.length === 3` |
| 5.7 | Check browser Network tab | `/api/exam/generate` returns valid `Exam` JSON | Response has `questions` array, each with `id`, `text`, `type`, `answer` |

### 6. Exam Answering + Submission

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| 6.1 | Answer MCQ questions by clicking radio buttons | Selected radio stays highlighted, other options in same group deselect | No flickering, state persists across re-renders |
| 6.2 | Answer free-text questions by typing in textarea | Text persists, no characters lost | Type 50+ chars, scroll away, scroll back — text still there |
| 6.3 | Click "Entregar examen" | Loading state, then results page | Submit button shows spinner, then navigates to results |
| 6.4 | Verify results page | Shows total score, per-question breakdown | Score displayed (e.g., "7/10"), each question has score + feedback |
| 6.5 | Verify evaluation feedback is in Spanish | All feedback text in Spanish | No English fragments in evaluator output |
| 6.6 | Check unanswered questions | Questions left blank show "No evaluable" or score 0 | Not all questions need answers, but unmarked ones handled gracefully |
| 6.7 | Check browser Network tab | `/api/evaluate` returns 200 with `EvaluationResult[]` | Each result has `score`, `feedback`, `is_evaluable` |

### 7. Profile Persistence + Dashboard

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| 7.1 | After completing exam (step 6), navigate to Dashboard | Stats cards show real numbers | Session count ≥ 1, Topics covered > 0, Average not "--%" |
| 7.2 | Verify StatsCards values | "Sesiones completadas" ≥ 1, "Temas cubiertos" ≥ 1, "Promedio general" shows percentage | All 4 cards have non-zero, non-placeholder values |
| 7.3 | Verify TopicChart | Bar chart renders with topic names and scores | At least 1 bar visible, topic labels readable |
| 7.4 | Verify WeakTopics | List of weak topics (score < 6) renders | If any topic score < 6, it appears here with score badge |
| 7.5 | Verify SessionHistory | Table/list shows session entries with date, questions, score | Entry for the session where exam was completed |
| 7.6 | Check browser Network tab | `/api/students/{id}/dashboard` returns `StudentProfile` with non-empty fields | `topicScores` has entries, `sessionHistory` has ≥ 1 entry |
| 7.7 | Empty state: new student with no exams | Dashboard shows "Todavía no tenés datos de progreso" | Clean empty state, no spinner forever, no 500 error |
| 7.8 | Check per-session profile | `/api/sessions/{id}/profile` returns scores filtered to that session | Response `topicScores` matches only exams in that session |

### 8. Exercise Generation

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| 8.1 | Type "generame un ejercicio práctico de agentes" | Exercise widget renders with problem statement + steps | Exercise card with structured content |
| 8.2 | Solve exercise and submit | Evaluation runs, results show | Score + feedback for exercise steps |
| 8.3 | Check profile updated | Dashboard reflects exercise completion | Topic scores for exercise topics updated |

### 9. Settings + Preferences

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| 9.1 | Navigate to Settings tab | Settings page loads with preference form | Difficulty selector, question type checkboxes, count input |
| 9.2 | Change difficulty to "Avanzado", question count to 3 | Settings saved | Confirmation toast or auto-save indicator |
| 9.3 | Generate exam with saved preferences | Exam respects difficulty and count | Questions match advanced difficulty, exactly 3 questions |

### 10. Error Recovery + Edge Cases

| # | Action | Expected Result | How to Verify |
|---|--------|----------------|----------------|
| 10.1 | Upload non-PDF file (e.g., `.exe` renamed to `.pdf`) | Ingestor rejects with clear error message | Error toast: "No se pudo procesar el archivo" or similar |
| 10.2 | Send empty chat message | Send button disabled or error toast | No 500 error, graceful handling |
| 10.3 | Kill backend during exam generation, restart | Frontend shows error state, not infinite spinner | Error message visible, app still responsive |
| 10.4 | Submit exam with all answers empty | Results show "No evaluable" or score 0 for all | No crash, results page renders |
| 10.5 | Rapid session switching (click 3 sessions fast) | UI stays responsive, no state corruption | Active session indicator follows clicks, no mixed file lists |
| 10.6 | Resize browser to mobile width (375px) | Layout adapts, sidebar collapses to hamburger | All elements still accessible, no horizontal scroll |

---

## LLM Agent Instructions

When an LLM agent runs this checklist, follow these rules:

1. **Use Playwright or browser automation** to execute each step — do not just read the page source.
2. **Assert deterministically**: check element visibility, text content, CSS classes, network responses.
3. **On failure**: capture screenshot, log the failing step number, note actual vs expected.
4. **Do not skip**: if a step cannot be executed (e.g., LLM provider down), mark it as `SKIP` with reason, continue remaining steps.
5. **Report format**: at end, produce a table with columns `Step | Status | Actual | Notes`.

### Example LLM Agent Report

```
| Step | Status | Actual | Notes |
|------|--------|--------|-------|
| S1   | PASS   | App loaded in 1.2s | — |
| S2   | PASS   | Session created, sidebar shows "Smoke Test" | — |
| S3   | PASS   | 15 temas detectados, 42 chunks | — |
| S4   | FAIL   | Only 2 questions rendered, expected 3 | LLM returned truncated response — see screenshot s4-fail.png |
| S5   | PASS   | Dashboard shows session count: 1 | — |
| 7.1  | FAIL   | Dashboard shows "--%" average, 0 temas | Profile not persisted after exam — bug MEMFIX-01 |
```

---

## Test Data

| File | Purpose | Location |
|------|---------|----------|
| `apunteAgentes_IA2007.pdf` | Primary test PDF (Spanish academic text, ~20 pages) | `back/tests/fixtures/` |
| Sample exam answers | Pre-written answers for deterministic evaluation testing | Create manually or use LLM |

---

## Version

- **Version**: 1.0
- **Last updated**: 2026-06-27
- **Covers**: Epic 9 (session lifecycle), Epic 11 (topic extraction), Epic 12 (memory/profile hardening + E2E)
