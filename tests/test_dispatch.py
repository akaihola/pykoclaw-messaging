"""Tests for the dispatch_to_agent helper."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from pykoclaw_messaging.dispatch import DispatchResult, dispatch_to_agent


@pytest.fixture()
def tmp_db(tmp_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        "CREATE TABLE IF NOT EXISTS conversations ("
        "    name TEXT PRIMARY KEY,"
        "    session_id TEXT,"
        "    cwd TEXT,"
        "    created_at TEXT NOT NULL"
        ");"
    )
    return db


def _make_agent_message(
    msg_type: str, text: str | None = None, session_id: str | None = None
) -> Any:
    from pykoclaw.agent_core import AgentMessage

    return AgentMessage(type=msg_type, text=text, session_id=session_id)


@pytest.mark.asyncio
async def test_dispatch_returns_concatenated_text(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    messages = [
        _make_agent_message("text", text="Hello"),
        _make_agent_message("text", text="World"),
        _make_agent_message("result", session_id="sess-1"),
    ]

    async def fake_query_agent(*_args: Any, **_kwargs: Any):  # noqa: ANN202
        for m in messages:
            yield m

    with patch("pykoclaw_messaging.dispatch.query_agent", fake_query_agent):
        result = await dispatch_to_agent(
            prompt="hi",
            channel_prefix="test",
            channel_id="123",
            db=tmp_db,
            data_dir=tmp_path,
        )

    assert isinstance(result, DispatchResult)
    assert result.full_text == "Hello\nWorld"
    assert result.session_id == "sess-1"


@pytest.mark.asyncio
async def test_dispatch_calls_on_text_for_each_chunk(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    messages = [
        _make_agent_message("text", text="chunk-a"),
        _make_agent_message("text", text="chunk-b"),
        _make_agent_message("result", session_id="sess-2"),
    ]

    async def fake_query_agent(*_args: Any, **_kwargs: Any):  # noqa: ANN202
        for m in messages:
            yield m

    on_text = AsyncMock()

    with patch("pykoclaw_messaging.dispatch.query_agent", fake_query_agent):
        result = await dispatch_to_agent(
            prompt="hi",
            channel_prefix="acp",
            channel_id="abc",
            db=tmp_db,
            data_dir=tmp_path,
            on_text=on_text,
        )

    assert on_text.await_count == 2
    on_text.assert_any_await("chunk-a")
    on_text.assert_any_await("chunk-b")
    assert result.full_text == "chunk-a\nchunk-b"


@pytest.mark.asyncio
async def test_dispatch_resumes_existing_session(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    tmp_db.execute(
        "INSERT INTO conversations (name, session_id, cwd, created_at)"
        " VALUES (?, ?, ?, ?)",
        ("wa-jid42", "old-sess", "/tmp", "2025-01-01T00:00:00"),
    )
    tmp_db.commit()

    captured_kwargs: dict[str, Any] = {}

    async def fake_query_agent(*_args: Any, **kwargs: Any):  # noqa: ANN202
        captured_kwargs.update(kwargs)
        yield _make_agent_message("result", session_id="new-sess")

    with patch("pykoclaw_messaging.dispatch.query_agent", fake_query_agent):
        result = await dispatch_to_agent(
            prompt="hi",
            channel_prefix="wa",
            channel_id="jid42",
            db=tmp_db,
            data_dir=tmp_path,
        )

    assert captured_kwargs["resume_session_id"] == "old-sess"
    assert captured_kwargs["conversation_name"] == "wa-jid42"
    assert result.session_id == "new-sess"


@pytest.mark.asyncio
async def test_dispatch_conversation_name_convention(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    captured_kwargs: dict[str, Any] = {}

    async def fake_query_agent(*_args: Any, **kwargs: Any):  # noqa: ANN202
        captured_kwargs.update(kwargs)
        yield _make_agent_message("result", session_id="s")

    with patch("pykoclaw_messaging.dispatch.query_agent", fake_query_agent):
        await dispatch_to_agent(
            prompt="hi",
            channel_prefix="tg",
            channel_id="999",
            db=tmp_db,
            data_dir=tmp_path,
            system_prompt="be helpful",
            model="claude-sonnet-4-20250514",
        )

    assert captured_kwargs["conversation_name"] == "tg-999"
    assert captured_kwargs["system_prompt"] == "be helpful"
    assert captured_kwargs["model"] == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_dispatch_no_result_message_keeps_resume_session_id(
    tmp_db: sqlite3.Connection, tmp_path: Path
) -> None:
    tmp_db.execute(
        "INSERT INTO conversations (name, session_id, cwd, created_at)"
        " VALUES (?, ?, ?, ?)",
        ("acp-x", "existing", "/tmp", "2025-01-01T00:00:00"),
    )
    tmp_db.commit()

    async def fake_query_agent(*_args: Any, **_kwargs: Any):  # noqa: ANN202
        yield _make_agent_message("text", text="only text")

    with patch("pykoclaw_messaging.dispatch.query_agent", fake_query_agent):
        result = await dispatch_to_agent(
            prompt="hi",
            channel_prefix="acp",
            channel_id="x",
            db=tmp_db,
            data_dir=tmp_path,
        )

    assert result.full_text == "only text"
    assert result.session_id == "existing"
