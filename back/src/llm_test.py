"""Deterministic mock LLM for E2E testing.

Activated by E2E_TEST_MODE=true. Matches prompts via SHA256 hash
to pre-recorded JSON seed files. In record mode (E2E_RECORD_MODE=true),
saves real LLM responses to seed files for later replay.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("ai-tutor.llm_test")

SEEDS_DIR = Path(__file__).parent.parent.parent / "front" / "e2e" / "fixtures"


class MockResponse:
    """Simulates an LLM response with optional tool calls."""

    def __init__(self, content: str, tool_calls: list[dict] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.response_metadata = {"mock": True}


class MockLLM:
    """Returns pre-recorded responses matched by prompt hash."""

    def __init__(self, seed_file: str):
        seed_path = SEEDS_DIR / seed_file
        if not seed_path.exists():
            logger.warning(
                "Seed file %s not found — mock will fail on first call. "
                "Run with E2E_RECORD_MODE=true to generate seeds.",
                seed_file,
            )
            self.seeds: dict[str, dict] = {}
        else:
            with open(seed_path) as f:
                seed_list: list[dict] = json.load(f)
            self.seeds = {s["prompt_hash"]: s for s in seed_list}
        self._record_mode = _is_record_mode()
        self._recorded: list[dict] = []

    @staticmethod
    def _normalize_message(m: Any) -> dict:
        """Convert a LangChain message or dict to a plain dict for hashing."""
        if isinstance(m, dict):
            return m
        if hasattr(m, "model_dump"):
            return m.model_dump()
        if hasattr(m, "type") and hasattr(m, "content"):
            kind = str(m.type)
            content = str(m.content) if m.content else ""
            result = {"role": kind, "content": content}
            if hasattr(m, "tool_calls") and m.tool_calls:
                result["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, "model_dump") else str(tc) for tc in m.tool_calls
                ]
            return result
        return {"content": str(m)}

    @classmethod
    def _normalize_messages(cls, messages: list) -> list[dict]:
        """Convert mixed message types to a list of plain dicts."""
        return [cls._normalize_message(m) for m in messages]

    async def ainvoke(self, messages: list[dict], **kwargs: Any) -> MockResponse:
        normalized = self._normalize_messages(messages)
        prompt_text = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:12]

        if self._record_mode:
            # In record mode, we can't actually make real calls here —
            # the override happens at the FastAPI level. This path
            # should not be hit directly.
            logger.warning("Record mode active but MockLLM.ainvoke called directly")

        # Try exact hash match first
        seed = self.seeds.get(prompt_hash)

        # Fallback: try label-based match (substring of prompt)
        if not seed:
            prompt_lower = prompt_text.lower()
            for s in self.seeds.values():
                label = s.get("label", "").lower()
                if label and label in prompt_lower:
                    seed = s
                    logger.debug("Label match: %s → %s", label, prompt_hash)
                    break

        if not seed:
            available_labels = [s.get("label", "?") for s in list(self.seeds.values())[:5]]
            raise ValueError(
                f"No seed for prompt hash {prompt_hash}. "
                f"Available labels: {available_labels}. "
                f"Run with E2E_RECORD_MODE=true to generate seeds."
            )

        return MockResponse(
            content=seed["response"],
            tool_calls=seed.get("tool_calls", []),
        )

    def invoke(self, prompt: Any, **kwargs: Any) -> MockResponse:
        """Sync invoke — wraps ainvoke in asyncio.run for LangChain compatibility."""
        import asyncio

        messages = prompt if isinstance(prompt, list) else [prompt]
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(messages, **kwargs))
        # Running loop exists (e.g. inside FastAPI) — use thread-safe approach
        import threading

        result_container: list[MockResponse] = []
        error_container: list[Exception] = []

        def _run():
            try:
                result_container.append(asyncio.run(self.ainvoke(messages, **kwargs)))
            except Exception as e:
                error_container.append(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join()
        if error_container:
            raise error_container[0]
        return result_container[0]

    async def ainvoke_with_config(self, *args: Any, **kwargs: Any) -> MockResponse:
        """Compatibility with LangChain's ainvoke pattern."""
        return await self.ainvoke(*args, **kwargs)


def _is_record_mode() -> bool:
    import os

    return os.getenv("E2E_RECORD_MODE", "").lower() == "true"


class RecordingLLM:
    """Wraps real LLM, records responses to seed file for later replay."""

    def __init__(self, real_llm: Any, seed_file: str = "recorded-seed.json"):
        self._real = real_llm
        self._seed_path = SEEDS_DIR / seed_file
        self._recorded: list[dict] = []
        # Load existing seeds if file exists (append mode)
        if self._seed_path.exists():
            with open(self._seed_path) as f:
                self._recorded = json.load(f)
            logger.info(
                "Loaded %d existing seed entries from %s",
                len(self._recorded),
                seed_file,
            )
        self._lock = _import_threading_lock()

    @staticmethod
    def _normalize_message(m: Any) -> dict:
        """Convert a LangChain message or dict to a plain dict for hashing."""
        if isinstance(m, dict):
            return m
        if hasattr(m, "model_dump"):
            return m.model_dump()
        if hasattr(m, "type") and hasattr(m, "content"):
            kind = str(m.type)
            content = str(m.content) if m.content else ""
            result = {"role": kind, "content": content}
            if hasattr(m, "tool_calls") and m.tool_calls:
                result["tool_calls"] = [
                    tc.model_dump() if hasattr(tc, "model_dump") else str(tc) for tc in m.tool_calls
                ]
            return result
        return {"content": str(m)}

    def _normalize_messages(self, messages: list) -> list[dict]:
        """Convert mixed message types to a list of plain dicts."""
        return [self._normalize_message(m) for m in messages]

    def _make_label(self, messages: list) -> str:
        """Generate a human-readable label from the first user message."""
        for m in messages:
            role = None
            content = ""
            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content", "")
            elif hasattr(m, "type"):
                role = str(m.type)
                content = str(m.content) if m.content else ""
            if role == "user" or role == "human":
                return content[:80].strip()
        return "unknown"

    def _make_response(self, entry: dict) -> MockResponse:
        """Create a MockResponse from a recorded entry."""
        return MockResponse(entry["response"], entry.get("tool_calls", []))

    def _hash_prompt(self, messages: list) -> str:
        normalized = self._normalize_messages(messages)
        prompt_text = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(prompt_text.encode()).hexdigest()[:12]

    async def ainvoke(self, messages: list[dict], **kwargs: Any) -> MockResponse:
        prompt_hash = self._hash_prompt(messages)

        # Check if already recorded (dedup)
        for existing in self._recorded:
            if existing.get("prompt_hash") == prompt_hash:
                logger.debug("Dedup hit for prompt hash %s", prompt_hash)
                return self._make_response(existing)

        # Call real LLM
        logger.info("Recording new prompt hash %s via real LLM", prompt_hash)
        response = await self._real.ainvoke(messages, **kwargs)

        return self._record_and_return(messages, prompt_hash, response)

    def invoke(self, prompt: Any, **kwargs: Any) -> MockResponse:
        """Sync invoke — wraps the real LLM's invoke and records response."""
        messages = prompt if isinstance(prompt, list) else [prompt]
        prompt_hash = self._hash_prompt(messages)

        # Check if already recorded (dedup)
        for existing in self._recorded:
            if existing.get("prompt_hash") == prompt_hash:
                logger.debug("Dedup hit for prompt hash %s", prompt_hash)
                return self._make_response(existing)

        # Call real LLM
        logger.info("Recording new prompt hash %s via real LLM (sync)", prompt_hash)
        response = self._real.invoke(prompt, **kwargs)

        return self._record_and_return(messages, prompt_hash, response)

    def _record_and_return(self, messages: list, prompt_hash: str, response: Any) -> MockResponse:

        # Extract content and tool_calls
        content = response.content if hasattr(response, "content") else str(response)
        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                if isinstance(tc, dict):
                    tool_calls.append(tc)
                elif hasattr(tc, "model_dump"):
                    tool_calls.append(tc.model_dump())
                else:
                    tool_calls.append(str(tc))

        # Record
        entry = {
            "prompt_hash": prompt_hash,
            "label": self._make_label(messages),
            "response": content,
            "tool_calls": tool_calls,
        }
        self._recorded.append(entry)

        # Save incrementally (thread-safe)
        with self._lock:
            self._seed_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._seed_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(self._recorded, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self._seed_path)

        logger.info(
            "Recorded seed entry %s (%d total)",
            prompt_hash,
            len(self._recorded),
        )
        return self._make_response(entry)

    async def ainvoke_with_config(self, *args: Any, **kwargs: Any) -> MockResponse:
        """Compatibility with LangChain's ainvoke pattern."""
        return await self.ainvoke(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped real LLM."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._real, name)

    @property
    def recorded_count(self) -> int:
        return len(self._recorded)


def _import_threading_lock():
    import threading

    return threading.Lock()


def get_mock_llm(seed_file: str = "recorded-seed.json") -> MockLLM:
    """Factory for E2E test mode. Returns MockLLM when E2E_TEST_MODE is true."""
    return MockLLM(seed_file)


def get_recording_llm(
    real_llm: Any,
    seed_file: str = "recorded-seed.json",
) -> RecordingLLM:
    """Factory for recording mode. Wraps a real LLM with response recording."""
    return RecordingLLM(real_llm, seed_file)
