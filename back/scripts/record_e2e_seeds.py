"""Record LLM responses for E2E mock seeds.

Usage:
    E2E_RECORD_MODE=true E2E_LIVE_LLM=true uv run python scripts/record_e2e_seeds.py

Runs key API calls against the running backend, which records LLM responses
to seed files. After recording, E2E tests can run in mock mode.
"""

import asyncio
import json
import os
from pathlib import Path

# Ensure E2E env vars are set
os.environ["E2E_RECORD_MODE"] = "true"
os.environ["E2E_LIVE_LLM"] = "true"


async def main():
    import httpx

    base_url = "http://localhost:8000"
    student_id = "e2e-test-student"

    async with httpx.AsyncClient(timeout=180.0) as client:
        # ── 1. Create session ──
        print("1. Creating session...")
        r = await client.post(
            f"{base_url}/api/sessions",
            json={"name": "E2E Seed Recording", "student_id": student_id},
        )
        r.raise_for_status()
        session = r.json()["data"]
        session_id = session["id"]
        print(f"   Session: {session_id}")

        # ── 2. Upload PDF ──
        print("2. Uploading PDF...")
        pdf_path = Path(__file__).parent.parent / "tests" / "fixtures" / "apunteAgentes_IA2007.pdf"
        if not pdf_path.exists():
            print(f"   ⚠️  PDF not found at {pdf_path}, skipping ingest")
        else:
            with open(pdf_path, "rb") as f:
                r = await client.post(
                    f"{base_url}/api/ingest",
                    files={"files": ("apunte.pdf", f, "application/pdf")},
                    data={"session_id": session_id},
                )
            r.raise_for_status()
            ingest = r.json()
            topics = ingest.get("data", [{}])[0].get("topicsDetected", [])
            print(f"   Topics: {len(topics)}")

        # ── 3. Generate exam (records classify_document + generate_exam) ──
        print("3. Generating exam...")
        r = await client.post(
            f"{base_url}/api/exam/generate",
            json={
                "session_id": session_id,
                "topic": "Agentes inteligentes",
                "question_count": 3,
                "question_types": ["mcq"],
                "studentId": student_id,
                "preferences": {
                    "difficulty": "medium",
                    "questionTypes": ["mcq"],
                    "questionCount": 3,
                    "includeTopics": [],
                    "excludeTopics": [],
                },
            },
        )
        r.raise_for_status()
        exam = r.json()["data"]
        exam_id = exam["id"]
        questions = exam["questions"]
        print(f"   Exam: {exam_id}, Questions: {len(questions)}")
        print(f"   topicDistribution: {exam.get('topicDistribution', {})}")

        # ── 4. Evaluate (records evaluate_answer) ──
        print("4. Evaluating...")
        answers = {}
        for q in questions:
            options = q.get("options", [])
            answers[q["id"]] = options[0] if options else "respuesta de prueba"

        r = await client.post(
            f"{base_url}/api/evaluate",
            json={
                "session_id": session_id,
                "exam_id": exam_id,
                "answers": answers,
                "examQuestions": questions,
                "studentId": student_id,
            },
        )
        r.raise_for_status()
        results = r.json()
        evals = results.get("data", [])
        scores = [e.get("score", "?") for e in evals]
        print(f"   Results: {len(evals)} evaluations, scores: {scores}")

        # ── 5. Verify dashboard ──
        print("5. Checking dashboard...")
        r = await client.get(f"{base_url}/api/students/{student_id}/dashboard")
        r.raise_for_status()
        dash = r.json()["data"]
        print(f"   sessionCount: {dash.get('sessionCount', '?')}")
        print(f"   topicScores: {list(dash.get('topicScores', {}).keys())}")
        print(f"   weakTopics: {dash.get('weakTopics', [])}")

        # ── Summary ──
        seed_file = (
            Path(__file__).parent.parent.parent
            / "front"
            / "e2e"
            / "fixtures"
            / "recorded-seed.json"
        )
        if seed_file.exists():
            with open(seed_file) as f:
                seeds = json.load(f)
            print(f"\n✅ Seeds recorded: {len(seeds)} entries → {seed_file}")
        else:
            print(f"\n⚠️  No seed file found at {seed_file}")
            print("   Check that E2E_RECORD_MODE=true was set when backend started.")


if __name__ == "__main__":
    asyncio.run(main())
