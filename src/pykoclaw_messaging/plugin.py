"""Messaging plugin — registers channel-agnostic CLI commands."""

from __future__ import annotations

import asyncio
import logging
from textwrap import dedent

import click

from pykoclaw.config import settings
from pykoclaw.db import enqueue_delivery, init_db
from pykoclaw.plugins import PykoClawPluginBase, load_plugins, run_db_migrations

from .dispatch import dispatch_to_agent

log = logging.getLogger(__name__)


def parse_conversation(conversation: str) -> tuple[str, str]:
    """Split ``"prefix-id"`` into ``(prefix, id)``.

    The prefix is everything before the first ``-``; the id is the rest
    (which may itself contain ``-``).

    Raises:
        ValueError: If the string contains no ``-`` separator.
    """
    sep = conversation.find("-")
    if sep < 1:
        raise ValueError(
            f"Invalid conversation name {conversation!r} — expected"
            " format 'prefix-id' (e.g. 'matrix-!room:server', 'wa-123@s.whatsapp.net')"
        )
    return conversation[:sep], conversation[sep + 1 :]


class MessagingPlugin(PykoClawPluginBase):
    """Registers the ``pykoclaw send`` command."""

    def register_commands(self, group: click.Group) -> None:
        @group.command()
        @click.argument("conversation")
        @click.argument("prompt")
        @click.option(
            "--no-deliver",
            is_flag=True,
            default=False,
            help="Run the agent but don't enqueue the reply for channel delivery.",
        )
        @click.option(
            "--model",
            default=None,
            help="Override the Claude model (e.g. 'claude-sonnet-4-20250514').",
        )
        def send(
            conversation: str,
            prompt: str,
            no_deliver: bool,
            model: str | None,
        ) -> None:
            """Send a one-off prompt to the agent and deliver the reply.

            CONVERSATION is the channel conversation name in 'prefix-id'
            format (e.g. 'matrix-!room:server', 'wa-123@s.whatsapp.net').

            PROMPT is the message to send to the agent.

            The agent's reply is printed to stdout and — unless --no-deliver
            is passed — enqueued for delivery to the channel.  The channel's
            'run' process must be active to pick up the delivery.
            """
            logging.basicConfig(
                level=logging.WARNING,
                format="%(asctime)s %(name)s %(levelname)s %(message)s",
            )

            try:
                channel_prefix, channel_id = parse_conversation(conversation)
            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise SystemExit(1) from None

            db = init_db(settings.db_path)

            plugins = load_plugins()
            run_db_migrations(db, plugins)

            click.echo(
                f"Dispatching to agent (conversation={conversation})...",
                err=True,
            )

            system_prompt = dedent("""\
                You are a helpful assistant. Your entire response will be \
                delivered as a message to a chat conversation. Write your \
                reply as if you are speaking directly to the recipient. \
                Do not include meta-commentary about sending or delivering \
                the message — just respond with the content itself.""")

            result = asyncio.run(
                dispatch_to_agent(
                    prompt=prompt,
                    channel_prefix=channel_prefix,
                    channel_id=channel_id,
                    db=db,
                    data_dir=settings.data,
                    model=model,
                    fresh=True,
                    system_prompt=system_prompt,
                )
            )

            if not result.full_text:
                click.echo("Agent produced no output.", err=True)
                raise SystemExit(0)

            # Always print the response to stdout.
            click.echo(result.full_text)

            if no_deliver:
                click.echo("(--no-deliver: skipping delivery queue)", err=True)
                return

            enqueue_delivery(
                db,
                task_id="send-cli",
                task_run_log_id=None,
                conversation=conversation,
                channel_prefix=channel_prefix,
                message=result.full_text,
            )
            click.echo(
                dedent(f"""\
                    ✓ Reply enqueued for delivery to {conversation}.
                      The channel's 'run' process will deliver it shortly."""),
                err=True,
            )
