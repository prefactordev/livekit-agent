# livekit-agent

Voice-first LiveKit web research agent with Exa-backed search and optional
Prefactor tracing.

## Setup

1. Copy the example environment file:

```sh
cp .env.example .env
```

2. Set:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `EXA_API_KEY`

3. Install dependencies:

```sh
mise run sync
```

4. Download the LiveKit model assets:

```sh
mise -E local run download-files
```

## Run

Run locally with console audio I/O:

```sh
mise -E local run console
```

Run against a LiveKit deployment:

```sh
mise -E local run dev
```

## Test

Run the offline test suite (no LiveKit, Exa or Prefactor calls):

```sh
mise run test
```

The turn-detector tests are skipped until the model files are downloaded
(`mise run download-files`).

## Optional Config

- `AGENT_PRESET` (`budget` or `balanced`)
- `EXA_SEARCH_MAX_RESULTS`
- `EXA_SEARCH_TYPE`
- `EXA_INCLUDE_DOMAINS`
- `PREFACTOR_API_URL`
- `PREFACTOR_API_TOKEN`
- `PREFACTOR_AGENT_ID`
- `PREFACTOR_AGENT_NAME`
- `LIVEKIT_REMOTE_EOT_URL`

See [.env.example](/Users/mcb/code/livekit-agent/.env.example).
