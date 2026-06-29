"""Tests for file metadata persistence (T-011..T-013)."""

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.memory.schema import (
    create_session,
    get_session,
    insert_ingested_document,
    list_session_files,
)


@pytest.fixture
async def db_session(tmp_path):
    """Yield an isolated in-memory DB session with migrations applied."""
    db_path = tmp_path / "files_persistence.db"
    with patch.object(settings, "sqlite_db_path", str(db_path)):
        from src.memory.schema import init_db

        await init_db()
        session_id = str(uuid.uuid4())
        await create_session(session_id, "Materia", "", session_id=session_id)
        yield session_id


class TestInsertIngestedDocument:
    """T-011: insert_ingested_document schema helper."""

    @pytest.mark.asyncio
    async def test_insert_persists_all_fields(self, db_session):
        """GIVEN a document dict → THEN row is inserted with all metadata."""
        session_id = db_session
        doc_id = str(uuid.uuid4())
        await insert_ingested_document(
            {
                "id": doc_id,
                "file_name": "calculo.pdf",
                "classification": "apunte_teorico",
                "topics_json": json.dumps(["derivadas", "límites"]),
                "chunks_count": 12,
                "session_id": session_id,
            }
        )

        rows = await list_session_files(session_id)
        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == doc_id
        assert row["file_name"] == "calculo.pdf"
        assert row["classification"] == "apunte_teorico"
        assert json.loads(row["topics_json"]) == ["derivadas", "límites"]
        assert row["chunks_count"] == 12
        assert row["session_id"] == session_id
        assert row["ingested_at"]

    @pytest.mark.asyncio
    async def test_insert_without_optional_fields(self, db_session):
        """GIVEN minimal doc → THEN defaults are applied."""
        session_id = db_session
        doc_id = str(uuid.uuid4())
        await insert_ingested_document(
            {
                "id": doc_id,
                "file_name": "fisica.txt",
                "session_id": session_id,
            }
        )

        rows = await list_session_files(session_id)
        assert len(rows) == 1
        assert rows[0]["chunks_count"] == 0
        assert rows[0]["classification"] is None

    @pytest.mark.asyncio
    async def test_insert_increments_session_file_count(self, db_session):
        """GIVEN two docs in same session → THEN get_session reports file_count 2."""
        session_id = db_session
        await insert_ingested_document(
            {
                "id": str(uuid.uuid4()),
                "file_name": "a.pdf",
                "session_id": session_id,
            }
        )
        await insert_ingested_document(
            {
                "id": str(uuid.uuid4()),
                "file_name": "b.pdf",
                "session_id": session_id,
            }
        )

        detail = await get_session(session_id)
        assert detail["file_count"] == 2


class TestListSessionFiles:
    """T-011: list_session_files ordering and filtering."""

    @pytest.mark.asyncio
    async def test_list_orders_by_recent_first(self, db_session):
        """GIVEN files uploaded at different times → THEN ordered DESC by ingested_at."""
        session_id = db_session
        ids = [str(uuid.uuid4()) for _ in range(3)]
        for idx, doc_id in enumerate(ids):
            await insert_ingested_document(
                {
                    "id": doc_id,
                    "file_name": f"file_{idx}.pdf",
                    "session_id": session_id,
                    "ingested_at": f"2026-06-26T10:00:0{idx}",
                }
            )

        rows = await list_session_files(session_id)
        assert [r["file_name"] for r in rows] == [
            "file_2.pdf",
            "file_1.pdf",
            "file_0.pdf",
        ]

    @pytest.mark.asyncio
    async def test_list_filters_by_session(self, db_session):
        """GIVEN files in two sessions → THEN only current session files returned."""
        session_a = db_session
        session_b = str(uuid.uuid4())
        await create_session(session_b, "Otra", "", session_id=session_b)

        await insert_ingested_document(
            {"id": str(uuid.uuid4()), "file_name": "a.pdf", "session_id": session_a}
        )
        await insert_ingested_document(
            {"id": str(uuid.uuid4()), "file_name": "b.pdf", "session_id": session_b}
        )

        rows = await list_session_files(session_a)
        assert len(rows) == 1
        assert rows[0]["file_name"] == "a.pdf"


class TestIngestEndpointPersistence:
    """T-012: /ingest router inserts file metadata after successful ingest."""

    def test_ingest_persists_metadata_after_success(self, tmp_path):
        """GIVEN successful ingest → THEN file row exists and is retrievable."""
        db_path = tmp_path / "ingest_router.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            from src.main import app
            from src.memory.schema import init_db

            asyncio.run(init_db())
            client = TestClient(app)
            doc_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())

            mock_tool = AsyncMock()
            mock_tool.ainvoke = AsyncMock(
                return_value={
                    "status": "ok",
                    "classification": "apunte_teorico",
                    "topics": ["derivadas"],
                    "chunks_created": 5,
                    "document_id": doc_id,
                }
            )
            with patch("src.tools.ingest_document", new=mock_tool):
                response = client.post(
                    "/api/ingest",
                    files=[("files", ("calculo.pdf", b"fake pdf", "application/pdf"))],
                    data={"session_id": session_id},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["error"] is None
        results = data["data"]
        assert results[0]["documentId"] == doc_id

        with patch.object(settings, "sqlite_db_path", str(db_path)):
            from src.memory.schema import init_db

            asyncio.run(init_db())
            rows = asyncio.run(list_session_files(session_id))
        assert len(rows) == 1
        assert rows[0]["file_name"] == "calculo.pdf"

    def test_ingest_creates_anonymous_session_row(self, tmp_path):
        """GIVEN no session_id → THEN session row + file row are created."""
        db_path = tmp_path / "ingest_anon.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            from src.main import app
            from src.memory.schema import init_db

            asyncio.run(init_db())
            client = TestClient(app)
            doc_id = str(uuid.uuid4())

            mock_tool = AsyncMock()
            mock_tool.ainvoke = AsyncMock(
                return_value={
                    "status": "ok",
                    "classification": "apunte_practico",
                    "topics": ["integrales"],
                    "chunks_created": 3,
                    "document_id": doc_id,
                }
            )
            with patch("src.tools.ingest_document", new=mock_tool):
                response = client.post(
                    "/api/ingest",
                    files=[("files", ("fisica.pdf", b"fake pdf", "application/pdf"))],
                )

        assert response.status_code == 200
        returned_session_id = response.json()["data"][0]["sessionId"]
        assert returned_session_id

        # File row should exist under the generated session id
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            from src.memory.schema import init_db

            asyncio.run(init_db())
            rows = asyncio.run(list_session_files(returned_session_id))
        assert len(rows) == 1
        assert rows[0]["file_name"] == "fisica.pdf"

    def test_ingest_failed_does_not_insert_row(self, tmp_path):
        """GIVEN ingest tool fails → THEN no file row is inserted."""
        db_path = tmp_path / "ingest_fail.db"
        with patch.object(settings, "sqlite_db_path", str(db_path)):
            from src.main import app
            from src.memory.schema import init_db

            asyncio.run(init_db())
            client = TestClient(app)
            session_id = str(uuid.uuid4())

            mock_tool = AsyncMock()
            mock_tool.ainvoke = AsyncMock(side_effect=RuntimeError("parse error"))
            with patch("src.tools.ingest_document", new=mock_tool):
                response = client.post(
                    "/api/ingest",
                    files=[("files", ("bad.pdf", b"fake", "application/pdf"))],
                    data={"session_id": session_id},
                )

        assert response.status_code == 200
        results = response.json()["data"]
        assert results[0]["status"] == "error"

        with patch.object(settings, "sqlite_db_path", str(db_path)):
            from src.memory.schema import init_db

            asyncio.run(init_db())
            rows = asyncio.run(list_session_files(session_id))
        assert rows == []


# ═══════════════════════════════════════════════════════════════════════════════
# TDR Phase — topic_descriptions_json persistence (TDR-05, TDR-06)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopicDescriptionsPersistence:
    """TDR-05: topic_descriptions_json column persisted and read back."""

    @pytest.mark.asyncio
    async def test_topic_descriptions_persisted_and_read(self, db_session):
        """TDR-05: Insert with descriptions → read them back."""
        session_id = db_session
        doc_id = str(uuid.uuid4())
        await insert_ingested_document(
            {
                "id": doc_id,
                "file_name": "agentes.pdf",
                "classification": "apunte_teorico",
                "topics_json": json.dumps(["Agentes inteligentes", "Razonamiento"]),
                "topic_descriptions_json": json.dumps(
                    {
                        "Agentes inteligentes": "Sistemas que perciben su entorno y actúan.",
                        "Razonamiento": "Proceso lógico para derivar conclusiones.",
                    }
                ),
                "chunks_count": 5,
                "session_id": session_id,
            }
        )

        rows = await list_session_files(session_id)
        assert len(rows) == 1
        row = rows[0]
        assert "topic_descriptions_json" in row
        descs = json.loads(row["topic_descriptions_json"])
        assert descs["Agentes inteligentes"] == "Sistemas que perciben su entorno y actúan."
        assert descs["Razonamiento"] == "Proceso lógico para derivar conclusiones."

    @pytest.mark.asyncio
    async def test_legacy_rows_no_descriptions_return_empty_dict(self, db_session):
        """TDR-05: Legacy row (no topic_descriptions_json) → returns {}."""
        session_id = db_session
        doc_id = str(uuid.uuid4())
        # Insert without topic_descriptions_json (legacy path)
        await insert_ingested_document(
            {
                "id": doc_id,
                "file_name": "legacy.pdf",
                "classification": "apunte_teorico",
                "topics_json": json.dumps(["Tema viejo"]),
                "chunks_count": 2,
                "session_id": session_id,
            }
        )

        rows = await list_session_files(session_id)
        assert len(rows) == 1
        desc_raw = rows[0].get("topic_descriptions_json")
        assert desc_raw is not None, "topic_descriptions_json should have default '{}'"
        descs = json.loads(desc_raw) if desc_raw else {}
        assert descs == {}

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self, db_session):
        """TDR-05: Malformed JSON in column → handled gracefully, falls back to {}."""
        import aiosqlite

        session_id = db_session
        doc_id = str(uuid.uuid4())
        # Insert valid doc first
        await insert_ingested_document(
            {
                "id": doc_id,
                "file_name": "corrupt.pdf",
                "session_id": session_id,
            }
        )
        # Manually corrupt the column
        async with aiosqlite.connect(settings.sqlite_db_path) as db:
            await db.execute(
                "UPDATE ingested_documents SET topic_descriptions_json = ? WHERE id = ?",
                ("not valid json", doc_id),
            )
            await db.commit()

        rows = await list_session_files(session_id)
        assert len(rows) == 1
        desc_raw = rows[0].get("topic_descriptions_json", "{}")
        # Should be parseable with a fallback
        try:
            descs = json.loads(desc_raw)
        except json.JSONDecodeError:
            descs = {}
        assert isinstance(descs, dict)

    @pytest.mark.asyncio
    async def test_topic_descriptions_column_migration_idempotent(self, db_session):
        """TDR-05: Migration is idempotent — calling twice doesn't break."""
        import aiosqlite

        from src.memory.schema import _column_exists, _run_migrations

        session_id = db_session
        async with aiosqlite.connect(settings.sqlite_db_path) as db:
            # Run migrations again (should be idempotent)
            await _run_migrations(db)
            await db.commit()
            # Column should exist
            assert await _column_exists(db, "ingested_documents", "topic_descriptions_json")

        # Still able to insert and read
        doc_id = str(uuid.uuid4())
        await insert_ingested_document(
            {
                "id": doc_id,
                "file_name": "migrate_test.pdf",
                "topic_descriptions_json": json.dumps({"Test": "Descripción de prueba."}),
                "session_id": session_id,
            }
        )
        rows = await list_session_files(session_id)
        assert len(rows) == 1
