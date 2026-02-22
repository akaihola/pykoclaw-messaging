"""Tests for the messaging plugin — parse_conversation + send CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from pykoclaw_messaging.plugin import MessagingPlugin, parse_conversation


# ── parse_conversation ────────────────────────────────────────────────


class TestParseConversation:
    def test_matrix_room(self) -> None:
        prefix, cid = parse_conversation("matrix-!QnMRhUnErgiTgBVTeY:matrix.org")
        assert prefix == "matrix"
        assert cid == "!QnMRhUnErgiTgBVTeY:matrix.org"

    def test_whatsapp_jid(self) -> None:
        prefix, cid = parse_conversation("wa-123456@s.whatsapp.net")
        assert prefix == "wa"
        assert cid == "123456@s.whatsapp.net"

    def test_acp_uuid(self) -> None:
        prefix, cid = parse_conversation("acp-a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert prefix == "acp"
        assert cid == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def test_simple(self) -> None:
        prefix, cid = parse_conversation("tg-999")
        assert prefix == "tg"
        assert cid == "999"

    def test_no_separator_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid conversation name"):
            parse_conversation("noprefix")

    def test_leading_dash_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid conversation name"):
            parse_conversation("-nope")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid conversation name"):
            parse_conversation("")


# ── send CLI command ──────────────────────────────────────────────────

# All patches target the names as imported into pykoclaw_messaging.plugin.
_P = "pykoclaw_messaging.plugin"


def _make_dispatch_result(text: str = "Agent says hello", session_id: str = "sess-1"):
    from pykoclaw_messaging.dispatch import DispatchResult

    return DispatchResult(full_text=text, session_id=session_id)


def _make_settings(tmp_path: Path) -> MagicMock:
    """Return a mock settings object pointing at a temp dir with a valid DB.

    Uses the real ``init_db`` to create the schema so it matches production.
    """
    from pykoclaw.db import init_db

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    db_path = data_dir / "pykoclaw.db"
    init_db(db_path)  # creates the real schema

    s = MagicMock()
    s.db_path = db_path
    s.data = data_dir
    return s


def _get_send_command() -> click.Command:
    group = click.Group()
    MessagingPlugin().register_commands(group)
    return group.commands["send"]


def test_send_invalid_conversation(tmp_path: Path) -> None:
    runner = CliRunner()
    cmd = _get_send_command()

    with patch(f"{_P}.settings", _make_settings(tmp_path)):
        result = runner.invoke(cmd, ["noprefix", "hello"])

    assert result.exit_code != 0
    assert "Invalid conversation name" in (result.output + (result.stderr or ""))


def test_send_dispatches_and_enqueues(tmp_path: Path) -> None:
    runner = CliRunner()
    cmd = _get_send_command()
    dispatch_result = _make_dispatch_result("Agent reply here")

    with (
        patch(f"{_P}.settings", _make_settings(tmp_path)),
        patch(
            f"{_P}.dispatch_to_agent",
            new_callable=AsyncMock,
            return_value=dispatch_result,
        ) as mock_dispatch,
        patch(f"{_P}.load_plugins", return_value=[]),
        patch(f"{_P}.run_db_migrations"),
        patch(f"{_P}.enqueue_delivery") as mock_enqueue,
    ):
        result = runner.invoke(cmd, ["matrix-!room:server", "What is 2+2?"])

    assert result.exit_code == 0, result.output
    assert "Agent reply here" in result.output

    # Verify dispatch was called with the right args
    mock_dispatch.assert_awaited_once()
    kw = mock_dispatch.call_args.kwargs
    assert kw["prompt"] == "What is 2+2?"
    assert kw["channel_prefix"] == "matrix"
    assert kw["channel_id"] == "!room:server"
    assert kw["fresh"] is True, "send must always start a fresh session"

    # Verify enqueue was called
    mock_enqueue.assert_called_once()
    eq_kw = mock_enqueue.call_args.kwargs
    assert eq_kw["conversation"] == "matrix-!room:server"
    assert eq_kw["channel_prefix"] == "matrix"
    assert eq_kw["message"] == "Agent reply here"


def test_send_no_deliver_skips_queue(tmp_path: Path) -> None:
    runner = CliRunner()
    cmd = _get_send_command()
    dispatch_result = _make_dispatch_result("dry run reply")

    with (
        patch(f"{_P}.settings", _make_settings(tmp_path)),
        patch(
            f"{_P}.dispatch_to_agent",
            new_callable=AsyncMock,
            return_value=dispatch_result,
        ),
        patch(f"{_P}.load_plugins", return_value=[]),
        patch(f"{_P}.run_db_migrations"),
        patch(f"{_P}.enqueue_delivery") as mock_enqueue,
    ):
        result = runner.invoke(cmd, ["wa-jid123", "test", "--no-deliver"])

    assert result.exit_code == 0, result.output
    assert "dry run reply" in result.output
    assert "--no-deliver" in result.output
    mock_enqueue.assert_not_called()


def test_send_empty_agent_output(tmp_path: Path) -> None:
    runner = CliRunner()
    cmd = _get_send_command()
    dispatch_result = _make_dispatch_result("")

    with (
        patch(f"{_P}.settings", _make_settings(tmp_path)),
        patch(
            f"{_P}.dispatch_to_agent",
            new_callable=AsyncMock,
            return_value=dispatch_result,
        ),
        patch(f"{_P}.load_plugins", return_value=[]),
        patch(f"{_P}.run_db_migrations"),
    ):
        result = runner.invoke(cmd, ["tg-42", "hello"])

    assert result.exit_code == 0
    assert "no output" in result.output.lower()


def test_send_passes_model_override(tmp_path: Path) -> None:
    runner = CliRunner()
    cmd = _get_send_command()
    dispatch_result = _make_dispatch_result("model reply")

    with (
        patch(f"{_P}.settings", _make_settings(tmp_path)),
        patch(
            f"{_P}.dispatch_to_agent",
            new_callable=AsyncMock,
            return_value=dispatch_result,
        ) as mock_dispatch,
        patch(f"{_P}.load_plugins", return_value=[]),
        patch(f"{_P}.run_db_migrations"),
        patch(f"{_P}.enqueue_delivery"),
    ):
        result = runner.invoke(
            cmd,
            ["matrix-!r:s", "hi", "--model", "claude-sonnet-4-20250514"],
        )

    assert result.exit_code == 0, result.output
    kw = mock_dispatch.call_args.kwargs
    assert kw["model"] == "claude-sonnet-4-20250514"
