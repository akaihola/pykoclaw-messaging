# pykoclaw-messaging

[![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-6f42c1?logo=anthropic&logoColor=white)](https://claude.ai/code)

> This project is developed by an AI coding agent ([Claude Code][claude-code]), with human oversight and direction.

Shared dispatch plugin/library for the public [pykoclaw core][pykoclaw]. This
repository is a separate support package inside the private `pykoclaw-dev` uv
workspace. It provides the channel-agnostic `dispatch_to_agent()` kernel used
by channel plugins and also registers the `pykoclaw send` CLI command through
the standard `pykoclaw.plugins` entry-point system.

## Features

- **Shared dispatch kernel** — resolves a conversation name from a channel
  prefix and channel ID, then calls `query_agent()`.
- **Session persistence** — resumes prior sessions through the core
  `conversations` table.
- **Resume recovery** — retries once with a fresh session if Claude session
  resumption fails with a `ProcessError`.
- **System-prompt invalidation** — starts a fresh session automatically when
  the stored system-prompt hash no longer matches the current prompt.
- **Streaming callback support** — plugins can stream partial text back to
  users while the agent is still responding.
- **Send CLI** — `pykoclaw send <conversation> <prompt>` runs a one-off
  dispatch and optionally enqueues the reply for channel delivery.

## Usage

```bash
pykoclaw send matrix-\!room:server "Summarize the latest discussion"
pykoclaw send wa-123456@s.whatsapp.net "Ping the group" --no-deliver
pykoclaw send slack-C01234567 "Status update" --model claude-sonnet-4-20250514
```

## Conversation naming

`pykoclaw send` expects a conversation in `prefix-id` form:

- `matrix-!room:server`
- `wa-123456@s.whatsapp.net`
- `acp-01234567-89ab-cdef-0123-456789abcdef`
- `slack-C01234567`

The prefix is everything before the first `-`; the rest is treated as the
channel-specific identifier.

## Delivery behaviour

By default, `pykoclaw send` both:

1. prints the agent reply to stdout, and
2. enqueues that reply into the core `delivery_queue` for channel delivery.

Pass `--no-deliver` to skip queueing and use it as a dry-run / local dispatch
command.

## Installation

```bash
uv tool install pykoclaw@git+https://github.com/akaihola/pykoclaw.git \
    --with=pykoclaw-messaging@git+https://github.com/akaihola/pykoclaw-messaging.git
```

Or with `uv pip install`:

```bash
uv pip install pykoclaw@git+https://github.com/akaihola/pykoclaw.git
uv pip install pykoclaw-messaging@git+https://github.com/akaihola/pykoclaw-messaging.git
```

See the [pykoclaw README][pykoclaw] for the full ecosystem overview.

[claude-code]: https://claude.ai/code
[pykoclaw]: https://github.com/akaihola/pykoclaw
