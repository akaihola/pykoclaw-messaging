"""Shared kernel for channel plugins: conversation lookup → query_agent() → result."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pykoclaw.agent_core import query_agent
from pykoclaw.db import DbConnection, get_conversation


@dataclass
class DispatchResult:
    full_text: str
    session_id: str | None


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

    Returns:
        A :class:`DispatchResult` with the concatenated response text and
        the session ID for future resumption.
    """
    conversation_name = f"{channel_prefix}-{channel_id}"

    conv = get_conversation(db, conversation_name)
    resume_session_id = conv.session_id if conv and conv.session_id else None

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
