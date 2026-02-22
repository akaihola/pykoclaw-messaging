"""Shared kernel for channel plugins: conversation lookup → query_agent() → result.

Handles two session-resume failure modes automatically:

1. **Stale system prompt** — the system prompt is baked into the session at
   creation.  If the prompt hash changes (e.g. after a code deploy), the old
   session is discarded and a fresh one is started.

2. **Corrupt session** — the Claude CLI subprocess crashes with
   ``ProcessError`` when resuming a corrupt/missing session.  On failure,
   the session is cleared and the call is retried once without resumption.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import ProcessError

from pykoclaw.agent_core import _prompt_hash, query_agent
from pykoclaw.db import DbConnection, get_conversation, upsert_conversation

log = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    full_text: str
    session_id: str | None


async def _run_agent(
    prompt: str,
    *,
    db: DbConnection,
    data_dir: Path,
    conversation_name: str,
    system_prompt: str | None,
    resume_session_id: str | None,
    extra_mcp_servers: dict[str, Any] | None,
    model: str | None,
    on_text: Callable[[str], Awaitable[None]] | None,
) -> DispatchResult:
    """Single attempt at running the agent.  Extracted so retry can re-call."""
    text_parts: list[str] = []
    session_id: str | None = resume_session_id

    async for msg in query_agent(
        prompt,
        db=db,
        data_dir=data_dir,
        conversation_name=conversation_name,
        system_prompt=system_prompt,
        resume_session_id=resume_session_id,
        extra_mcp_servers=extra_mcp_servers,
        model=model,
    ):
        if msg.type == "text" and msg.text:
            text_parts.append(msg.text)
            if on_text is not None:
                await on_text(msg.text)
        elif msg.type == "result":
            session_id = msg.session_id
            # Use ResultMessage.result as fallback when no TextBlock
            # messages were streamed (can happen during multi-turn tool
            # use where the final text only appears in the result).
            if msg.text and not text_parts:
                text_parts.append(msg.text)
                if on_text is not None:
                    await on_text(msg.text)

    return DispatchResult(
        full_text="\n".join(text_parts).strip(),
        session_id=session_id,
    )


async def dispatch_to_agent(
    *,
    prompt: str,
    channel_prefix: str,
    channel_id: str,
    db: DbConnection,
    data_dir: Path,
    system_prompt: str | None = None,
    extra_mcp_servers: dict[str, Any] | None = None,
    model: str | None = None,
    on_text: Callable[[str], Awaitable[None]] | None = None,
    fresh: bool = False,
) -> DispatchResult:
    """Send *prompt* to the agent on behalf of a channel conversation.

    Args:
        prompt: The user message to send.
        channel_prefix: Short lowercase prefix (``"wa"``, ``"acp"``,
            ``"tg"``).  Combined with *channel_id* to form the conversation
            name ``"{prefix}-{id}"``.
        channel_id: Channel-specific conversation identifier (JID, session
            UUID, Telegram chat ID, …).
        db: Database connection (core ``conversations`` table must exist).
        data_dir: pykoclaw data directory (``settings.data``).
        system_prompt: Optional system prompt forwarded to the agent.
        extra_mcp_servers: Additional MCP server definitions injected by the
            channel plugin (e.g. ``send_message`` for WhatsApp).
        model: Override the default Claude model for this call.
        on_text: Async callback invoked for every text chunk as it streams
            from the agent.  Use this for real-time delivery (e.g. ACP
            ``session/update`` notifications, Telegram ``send_message``).
        fresh: If ``True``, start a new session instead of resuming an
            existing one.  The conversation name is still used for the
            working directory and DB lookup, but the session is not resumed.

    Returns:
        A :class:`DispatchResult` with the concatenated response text and
        the session ID for future resumption.
    """
    conversation_name = f"{channel_prefix}-{channel_id}"

    if fresh:
        resume_session_id = None
    else:
        conv = get_conversation(db, conversation_name)
        resume_session_id = conv.session_id if conv and conv.session_id else None

        # Invalidate session if system prompt changed since creation.
        if resume_session_id and conv and system_prompt:
            current_hash = _prompt_hash(system_prompt)
            if conv.system_prompt_hash and conv.system_prompt_hash != current_hash:
                log.info(
                    "System prompt changed for %s — starting fresh session",
                    conversation_name,
                )
                resume_session_id = None

    common = dict(
        db=db,
        data_dir=data_dir,
        conversation_name=conversation_name,
        system_prompt=system_prompt,
        extra_mcp_servers=extra_mcp_servers,
        model=model,
        on_text=on_text,
    )

    try:
        return await _run_agent(prompt, resume_session_id=resume_session_id, **common)
    except ProcessError:
        if resume_session_id is None:
            raise  # already fresh — nothing to retry
        log.warning(
            "Session resume failed for %s (session=%s) — retrying fresh",
            conversation_name,
            resume_session_id,
        )
        # Clear the stale session so future calls don't hit the same error.
        upsert_conversation(db, conversation_name, "", str(data_dir))
        return await _run_agent(prompt, resume_session_id=None, **common)
