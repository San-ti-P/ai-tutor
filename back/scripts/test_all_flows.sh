#!/usr/bin/env bash
# Test all 7 orchestrator intents via curl. Requires backend running:
#   cd back && uv run uvicorn src.main:app --reload
# Run: bash back/scripts/test_all_flows.sh

BASE="http://localhost:8000/api"
SID="curl-test-$(date +%s)"

echo "=== Testing AI Tutor — all orchestrator flows ==="
echo "Session: $SID"
echo

# 1. general_chat
echo "--- 1. general_chat ---"
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Hola, ¿cómo estás?\", \"session_id\": \"$SID\"}" | python -m json.tool 2>/dev/null || echo "(raw output above)"
echo

# 2. generate_exam
echo "--- 2. generate_exam ---"
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Generame un examen sobre agentes inteligentes con 5 preguntas\", \"session_id\": \"$SID\"}" | python -m json.tool 2>/dev/null || echo "(raw output above)"
echo

# 3. generate_exercise
echo "--- 3. generate_exercise ---"
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Dame un ejercicio práctico sobre agentes inteligentes\", \"session_id\": \"$SID\"}" | python -m json.tool 2>/dev/null || echo "(raw output above)"
echo

# 4. ingest intent
echo "--- 4. ingest ---"
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Quiero subir un apunte sobre agentes inteligentes\", \"session_id\": \"$SID\"}" | python -m json.tool 2>/dev/null || echo "(raw output above)"
echo

# 5. evaluate
echo "--- 5. evaluate ---"
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Necesito que evalúes mis respuestas del examen\", \"session_id\": \"$SID\"}" | python -m json.tool 2>/dev/null || echo "(raw output above)"
echo

# 6. query_profile
echo "--- 6. query_profile ---"
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"¿Cómo va mi progreso? Mostrame mi perfil\", \"session_id\": \"$SID\"}" | python -m json.tool 2>/dev/null || echo "(raw output above)"
echo

# 7. composite (multi-step)
echo "--- 7. composite ---"
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Primero mostrame mi perfil, después generame un examen y dame un ejercicio práctico\", \"session_id\": \"$SID\"}" | python -m json.tool 2>/dev/null || echo "(raw output above)"
echo

echo "=== Done ==="
echo "Check https://us.cloud.langfuse.com for session: $SID"
