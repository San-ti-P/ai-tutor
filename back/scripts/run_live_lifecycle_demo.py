"""Live lifecycle integration demo: ingestion -> exam generation -> evaluation -> profile metrics.

Requires environment variables set in .env (e.g. LLM_PROVIDER=ollama with api key, Langfuse keys).
Run from back/:
    uv run python scripts/run_live_lifecycle_demo.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

# Add back/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Force live LLM mode for the simulation/demo
os.environ["E2E_LIVE_LLM"] = "true"
os.environ["E2E_TEST_MODE"] = "false"

from fastapi.testclient import TestClient
from src.main import app

def main():
    with TestClient(app) as client:
        # 1. Create a unique session ID
        session_id = str(uuid.uuid4())
        print(f"=== Starting Session Lifecycle Demo ===")
        print(f"Session ID: {session_id}")
        print()

        # 2. Check initial session info
        print("--- Checking initial session status ---")
        resp = client.get(f"/api/sessions/{session_id}")
        if resp.status_code == 404:
            print("Session does not exist yet (expected).")
        else:
            print(f"Session info: {resp.json()}")
        print()

        # 3. Ingest the academic PDF apunteAgentes_IA2007.pdf
        pdf_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "apunteAgentes_IA2007.pdf"
        if not pdf_path.exists():
            print(f"ERROR: PDF file not found at {pdf_path}")
            return

        print(f"--- Ingesting apunteAgentes_IA2007.pdf ---")
        with open(pdf_path, "rb") as f:
            files = [("files", (pdf_path.name, f, "application/pdf"))]
            data = {"session_id": session_id}
            resp = client.post("/api/ingest", files=files, data=data)
        
        assert resp.status_code == 200, f"Ingestion failed: {resp.text}"
        ingest_data = resp.json()["data"][0]
        print(f"Ingestion Succeeded!")
        print(f"Classification: {ingest_data.get('classification')}")
        print(f"Topics Detected: {ingest_data.get('topicsDetected')}")
        print(f"Chunks Created: {ingest_data.get('chunksCreated')}")
        print()

        # 4. Verify session status and file_count are updated
        print("--- Verifying session state after ingestion ---")
        resp = client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 200, f"Failed to get session details: {resp.text}"
        session_data = resp.json()["data"]
        print(f"Session Status: {session_data['status']} (Expected: active)")
        print(f"File Count: {session_data['fileCount']} (Expected: 1)")
        print(f"Exam Count: {session_data['examCount']} (Expected: 0)")
        assert session_data["status"] == "active"
        assert session_data["fileCount"] == 1
        print()

        # 5. Generate an exam based on one of the topics detected (fallback to "Agentes Inteligentes")
        topic = "Agentes Inteligentes"
        if ingest_data.get('topicsDetected'):
            topic = ingest_data['topicsDetected'][0]
        
        print(f"--- Generating Exam for topic: '{topic}' ---")
        exam_payload = {
            "session_id": session_id,
            "topic": topic,
            "preferences": {
                "questionCount": 3,
                "questionTypes": ["mcq", "open"],
                "difficulty": "medium",
                "includeTopics": [],
                "excludeTopics": []
            }
        }
        resp = client.post("/api/exam/generate", json=exam_payload)
        assert resp.status_code == 200, f"Exam generation failed: {resp.text}"
        exam_data = resp.json()["data"]
        exam_id = exam_data["id"]
        questions = exam_data["questions"]
        print(f"Exam Succeeded!")
        print(f"Exam ID: {exam_id}")
        print(f"Questions Generated: {len(questions)}")
        for i, q in enumerate(questions):
            print(f"  Q{i+1}: [{q['type']}] {q['prompt'][:80]}...")
        print()

        # 6. Verify session exam_count updated to 1
        print("--- Verifying session state after exam generation ---")
        resp = client.get(f"/api/sessions/{session_id}")
        session_data = resp.json()["data"]
        print(f"Exam Count: {session_data['examCount']} (Expected: 1)")
        assert session_data["examCount"] == 1
        print()

        # 7. Evaluate mock answers for the generated exam
        print("--- Submitting exam evaluations ---")
        answers = {}
        exam_questions = []
        for q in questions:
            # Build question map
            exam_questions.append({
                "id": q["id"],
                "prompt": q["prompt"],
                "type": q["type"],
                "base_answer": q.get("baseAnswer", "Respuesta correcta por defecto"),
                "options": q.get("options", []),
                "source_chunk_ids": q.get("sourceChunkIds", []),
                "topic": q.get("topic", topic),
                "difficulty": q.get("difficulty", "medium")
            })
            # Simulate student answer (simulate correct answers)
            if q["type"] == "mcq":
                answers[q["id"]] = q.get("baseAnswer", "Opción A")
            else:
                answers[q["id"]] = "Los agentes inteligentes son entidades que perciben su entorno y actúan sobre él racionalmente."

        eval_payload = {
            "session_id": session_id,
            "exam_id": exam_id,
            "answers": answers,
            "exam_questions": exam_questions
        }
        resp = client.post("/api/evaluate", json=eval_payload)
        assert resp.status_code == 200, f"Evaluation failed: {resp.text}"
        eval_results = resp.json()["data"]
        print("Evaluation Results:")
        for res in eval_results:
            print(f"  Q ID: {res['questionId']}")
            print(f"    Score: {res['score']}")
            print(f"    Justification: {res.get('justification', '')[:100]}...")
        print()

        # 8. Check updated average score in session profile
        print("--- Retrieving Session Profile metrics ---")
        resp = client.get(f"/api/sessions/{session_id}/profile")
        assert resp.status_code == 200, f"Failed to get session profile: {resp.text}"
        profile_data = resp.json()["data"]
        print(f"Profile Exam Count: {profile_data['examCount']} (Expected: 1)")
        print(f"Profile Average Score: {profile_data['averageScore']}")
        print(f"Profile Weak Topics: {profile_data['weakTopics']}")
        print(f"Profile Topic Scores: {profile_data['topicScores']}")
        assert profile_data["examCount"] == 1
        assert profile_data["averageScore"] is not None
        print()

        # 9. Verify traces in Langfuse using langfuse-cli
        print("--- Querying Langfuse CLI for verification ---")
        os.system(f"LANGFUSE_PUBLIC_KEY=\"pk-lf-e9456663-7b3d-4b43-aa1d-6431a31935ac\" LANGFUSE_SECRET_KEY=\"sk-lf-2df816f5-bab7-45ff-b557-95782682fc8f\" LANGFUSE_HOST=\"https://us.cloud.langfuse.com\" npx langfuse-cli api traces list --session-id {session_id} --limit 5 --json")

        print("\n=== Session Lifecycle Demo Completed Successfully! ===")

if __name__ == "__main__":
    main()
