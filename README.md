# livekit-agent

Voice-first LiveKit web research agent with Exa-backed web search and optional Prefactor tracing.

## What it does

- Runs a LiveKit agent that searches the web before answering time-sensitive questions.
- Supports a text-only smoke path for quick validation without full voice I/O.
- Can emit Prefactor traces when Prefactor credentials are configured.

## Requirements

- Python 3.12
- `uv`
- `mise`
- LiveKit credentials
- Exa API key

## Setup

1. Create local env config from the example values you want to use.
2. Put secrets in `mise.local.toml` or a local `.env` file.
3. Install dependencies:

```sh
mise run sync
```

## Environment

Required:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `EXA_API_KEY`

Optional:

- `AGENT_PRESET` (`budget` or `balanced`)
- `EXA_SEARCH_MAX_RESULTS`
- `EXA_SEARCH_TYPE`
- `EXA_INCLUDE_DOMAINS`
- `PREFACTOR_API_URL`
- `PREFACTOR_API_TOKEN`
- `PREFACTOR_AGENT_ID`
- `PREFACTOR_AGENT_NAME`
- `LIVEKIT_REMOTE_EOT_URL`

See [`/Users/mcb/code/livekit-agent/.env.example`](/Users/mcb/code/livekit-agent/.env.example) for the expected variables.

## Run

Text-only smoke test:

```sh
mise -E local run smoke-text
```

Console audio I/O:

```sh
mise -E local run console
```

Against a LiveKit deployment:

```sh
mise -E local run dev
```

## Prefactor tracing

Prefactor tracing is enabled only when both `PREFACTOR_API_URL` and `PREFACTOR_API_TOKEN` are set. If tracing is enabled, `PREFACTOR_AGENT_ID` must point to a real Prefactor agent.

To verify that expected Prefactor spans are emitted:

```sh
mise -E local run verify-prefactor
```

## Turn detector assets

For `console`, `dev`, and `start`, the agent expects local LiveKit multilingual turn-detector assets unless `LIVEKIT_REMOTE_EOT_URL` is set. If assets are missing, the runtime will tell you to download them with:

```sh
uv run python main.py download-files
```

## Repo notes

- [`/Users/mcb/code/livekit-agent/mise.local.toml`](/Users/mcb/code/livekit-agent/mise.local.toml) is intentionally ignored because it contains local secrets.
- Virtualenvs, caches, and generated Python artifacts are ignored via [`/Users/mcb/code/livekit-agent/.gitignore`](/Users/mcb/code/livekit-agent/.gitignore).
