"""
Voice-first LiveKit web research agent.

The agent uses Exa for web search and can optionally emit Prefactor traces
through the published `prefactor-livekit` package.

Setup:
  # required env vars:
  #   LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, EXA_API_KEY
  # optional Prefactor env vars:
  #   PREFACTOR_API_URL, PREFACTOR_API_TOKEN, PREFACTOR_AGENT_ID, PREFACTOR_AGENT_NAME

Run:
  mise -E local run smoke-text
  mise -E local run console
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from urllib.parse import urlparse

from dotenv import load_dotenv
from exa_py import Exa
from exa_py.api import Result, SearchResponse
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from prefactor_livekit import PrefactorLiveKitSession

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_PRESET = "budget"
DEFAULT_PREFACTOR_AGENT_ID = "web-research-agent"
DEFAULT_PREFACTOR_AGENT_NAME = "Web Research Agent"
DEFAULT_PREFACTOR_DOWNLOAD_COMMAND = "mise -E local run download-files"
DEFAULT_EXA_MAX_RESULTS = 5
DEFAULT_EXA_SEARCH_TYPE = "auto"
MAX_EXA_RESULTS = 5
SEARCH_RESULT_TEXT_CHAR_LIMIT = 1600
VALID_EXA_SEARCH_TYPES = frozenset(
    {
        "neural",
        "keyword",
        "auto",
        "hybrid",
        "fast",
        "deep-reasoning",
        "deep-max",
        "deep-lite",
        "magic",
        "deep",
        "instant",
    }
)
TURN_DETECTOR_REMOTE_ENV = "LIVEKIT_REMOTE_EOT_URL"


@dataclass(frozen=True)
class AgentConfig:
    preset: str
    llm_model: str
    stt_model: str
    stt_language: str
    tts_model: str
    tts_voice: str


@dataclass(frozen=True)
class SearchConfig:
    api_key: str
    max_results: int
    search_type: str
    include_domains: tuple[str, ...]


PRESET_CONFIGS: dict[str, AgentConfig] = {
    "budget": AgentConfig(
        preset="budget",
        llm_model="openai/gpt-5-mini",
        stt_model="deepgram/flux-general",
        stt_language="en",
        tts_model="deepgram/aura-2",
        tts_voice="athena",
    ),
    "balanced": AgentConfig(
        preset="balanced",
        llm_model="openai/gpt-5.4",
        stt_model="deepgram/nova-3",
        stt_language="en",
        tts_model="cartesia/sonic-3",
        tts_voice="a4a16c5e-5902-4732-b9b6-2a48efd2e11b",
    ),
}


def resolve_agent_config(env: Mapping[str, str] | None = None) -> AgentConfig:
    resolved_env = os.environ if env is None else env
    preset_name = (
        resolved_env.get("AGENT_PRESET", DEFAULT_PRESET).strip() or DEFAULT_PRESET
    )
    base_config = PRESET_CONFIGS.get(preset_name)
    if base_config is None:
        supported = ", ".join(sorted(PRESET_CONFIGS))
        raise ValueError(
            f"Unsupported AGENT_PRESET '{preset_name}'. Expected one of: {supported}"
        )

    return base_config


def _require_env(name: str, env: Mapping[str, str]) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _parse_positive_int(
    name: str,
    raw_value: str | None,
    *,
    default: int,
    max_value: int,
) -> int:
    value = (raw_value or "").strip()
    if not value:
        return default

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    if parsed > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return parsed


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _normalize_domains(domains: list[str] | None) -> list[str] | None:
    if domains is None:
        return None
    normalized = [item.strip() for item in domains if item.strip()]
    return normalized or None


def _resolve_search_type(search_type: str, fallback: str) -> str:
    normalized = search_type.strip() or fallback
    if normalized not in VALID_EXA_SEARCH_TYPES:
        return fallback
    return normalized


def resolve_search_config(env: Mapping[str, str] | None = None) -> SearchConfig:
    resolved_env = os.environ if env is None else env
    api_key = _require_env("EXA_API_KEY", resolved_env)
    max_results = _parse_positive_int(
        "EXA_SEARCH_MAX_RESULTS",
        resolved_env.get("EXA_SEARCH_MAX_RESULTS"),
        default=DEFAULT_EXA_MAX_RESULTS,
        max_value=MAX_EXA_RESULTS,
    )
    search_type = (
        resolved_env.get("EXA_SEARCH_TYPE", DEFAULT_EXA_SEARCH_TYPE).strip()
        or DEFAULT_EXA_SEARCH_TYPE
    )
    if search_type not in VALID_EXA_SEARCH_TYPES:
        supported = ", ".join(sorted(VALID_EXA_SEARCH_TYPES))
        raise ValueError(f"EXA_SEARCH_TYPE must be one of: {supported}")
    include_domains = _parse_csv(resolved_env.get("EXA_INCLUDE_DOMAINS"))
    return SearchConfig(
        api_key=api_key,
        max_results=max_results,
        search_type=search_type,
        include_domains=include_domains,
    )


def build_session(
    config: AgentConfig,
    *,
    vad: Any | None = None,
    turn_detection: Any | None = None,
) -> AgentSession:
    session_kwargs: dict[str, Any] = {
        "llm": inference.LLM(config.llm_model),
        "stt": inference.STT(model=config.stt_model, language=config.stt_language),
        "tts": inference.TTS(model=config.tts_model, voice=config.tts_voice),
        "preemptive_generation": True,
    }
    if vad is not None:
        session_kwargs["vad"] = vad
    if turn_detection is not None:
        session_kwargs["turn_handling"] = {"turn_detection": turn_detection}

    return AgentSession(**session_kwargs)


def build_exa_client(config: SearchConfig) -> Exa:
    return Exa(config.api_key)


def _extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return hostname.lower() or None


def _trim_text(value: str | None, *, limit: int = SEARCH_RESULT_TEXT_CHAR_LIMIT) -> str:
    if not value:
        return ""
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def normalize_search_result(result: Result) -> dict[str, Any]:
    return {
        "title": result.title,
        "url": result.url,
        "domain": _extract_domain(result.url),
        "published_date": result.published_date,
        "author": result.author,
        "text": _trim_text(result.text),
    }


def normalize_search_response(
    *,
    query: str,
    response: SearchResponse[Result],
) -> dict[str, Any]:
    normalized_results = [
        normalize_search_result(result) for result in response.results
    ]
    return {
        "query": query,
        "resolved_search_type": response.resolved_search_type,
        "search_time_ms": response.search_time,
        "result_count": len(normalized_results),
        "results": normalized_results,
    }


def run_exa_search(
    client: Exa,
    search_config: SearchConfig,
    *,
    query: str,
    num_results: int = DEFAULT_EXA_MAX_RESULTS,
    search_type: str = DEFAULT_EXA_SEARCH_TYPE,
    include_domains: list[str] | None = None,
) -> dict[str, Any]:
    resolved_num_results = min(max(1, num_results), search_config.max_results)
    resolved_search_type = _resolve_search_type(search_type, search_config.search_type)
    resolved_domains = _normalize_domains(include_domains)
    if include_domains is None:
        resolved_domains = list(search_config.include_domains) or None
    response = client.search(
        query,
        contents={"text": True},
        num_results=resolved_num_results,
        include_domains=resolved_domains,
        type=resolved_search_type,
    )
    return normalize_search_response(query=query, response=response)


def build_prefactor_tool_schemas() -> dict[str, dict[str, Any]]:
    return {
        "search_web": {
            "span_type": "search_web",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "minimum": 1},
                    "search_type": {
                        "type": "string",
                        "enum": sorted(VALID_EXA_SEARCH_TYPES),
                    },
                    "include_domains": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "result_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "resolved_search_type": {"type": ["string", "null"]},
                    "search_time_ms": {"type": ["number", "null"]},
                    "result_count": {"type": "integer"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": ["string", "null"]},
                                "url": {"type": ["string", "null"]},
                                "domain": {"type": ["string", "null"]},
                                "published_date": {"type": ["string", "null"]},
                                "author": {"type": ["string", "null"]},
                                "text": {"type": "string"},
                            },
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["query", "result_count", "results"],
                "additionalProperties": False,
            },
        }
    }


def build_prefactor_tracer(
    env: Mapping[str, str] | None = None,
) -> PrefactorLiveKitSession | None:
    resolved_env = os.environ if env is None else env
    api_url = (resolved_env.get("PREFACTOR_API_URL") or "").strip()
    api_token = (resolved_env.get("PREFACTOR_API_TOKEN") or "").strip()
    if not api_url or not api_token:
        return None

    agent_id = (resolved_env.get("PREFACTOR_AGENT_ID") or "").strip()
    if not agent_id:
        raise ValueError(
            "PREFACTOR_AGENT_ID is required when Prefactor tracing is enabled. "
            "Set it to the real Prefactor agent ID you want to inspect."
        )
    agent_name = (
        resolved_env.get("PREFACTOR_AGENT_NAME", DEFAULT_PREFACTOR_AGENT_NAME).strip()
        or DEFAULT_PREFACTOR_AGENT_NAME
    )
    logger.info(
        "Prefactor tracing enabled for agent_id=%s agent_name=%s",
        agent_id,
        agent_name,
    )
    return PrefactorLiveKitSession.from_config(
        api_url=api_url,
        api_token=api_token,
        agent_id=agent_id,
        agent_name=agent_name,
        tool_schemas=build_prefactor_tool_schemas(),
    )


def _resolve_close_reason(reason: Any) -> str:
    resolved = getattr(reason, "value", reason)
    if isinstance(resolved, str):
        return resolved
    return "unknown"


def ensure_turn_detector_assets(env: Mapping[str, str] | None = None) -> None:
    resolved_env = os.environ if env is None else env
    if (resolved_env.get(TURN_DETECTOR_REMOTE_ENV) or "").strip():
        return

    from huggingface_hub import hf_hub_download
    from livekit.plugins.turn_detector.models import (
        HG_MODEL,
        MODEL_REVISIONS,
        ONNX_FILENAME,
    )
    from transformers import AutoTokenizer

    revision = MODEL_REVISIONS["multilingual"]
    missing_assets: list[str] = []

    def _check_hf_file(filename: str, *, subfolder: str | None = None) -> None:
        try:
            download_kwargs: dict[str, Any] = {
                "repo_id": HG_MODEL,
                "filename": filename,
                "revision": revision,
                "local_files_only": True,
            }
            if subfolder is not None:
                download_kwargs["subfolder"] = subfolder
            hf_hub_download(**download_kwargs)
        except Exception:
            if subfolder is None:
                missing_assets.append(filename)
            else:
                missing_assets.append(f"{subfolder}/{filename}")

    _check_hf_file("languages.json")
    _check_hf_file(ONNX_FILENAME, subfolder="onnx")
    try:
        AutoTokenizer.from_pretrained(
            HG_MODEL,
            revision=revision,
            local_files_only=True,
            truncation_side="left",
        )
    except Exception:
        missing_assets.append("tokenizer")

    if not missing_assets:
        return

    unique_assets = ", ".join(dict.fromkeys(missing_assets))
    raise RuntimeError(
        "Missing local LiveKit turn-detector assets for "
        f"{HG_MODEL}@{revision}: {unique_assets}. "
        f"Run `{DEFAULT_PREFACTOR_DOWNLOAD_COMMAND}` or "
        "`uv run python main.py download-files` from this directory."
    )


def should_validate_turn_detector(argv: list[str] | None = None) -> bool:
    args = sys.argv[1:] if argv is None else argv
    command = next((arg for arg in args if not arg.startswith("-")), None)
    return command in {"console", "dev", "start"}


async def start_session(
    *,
    session: AgentSession,
    agent: Agent,
    tracer: PrefactorLiveKitSession | None,
    **start_kwargs: Any,
) -> str | None:
    if tracer is None:
        await session.start(agent=agent, **start_kwargs)
        return None

    try:
        instance = await tracer.ensure_initialized()
        logger.info(
            "Prefactor instance initialized for agent_id=%s instance_id=%s",
            tracer._agent_id,
            instance.id,
        )
        await tracer.start(
            session=session,
            agent=agent,
            **start_kwargs,
        )
        return instance.id
    except Exception:
        logger.exception("Prefactor tracing startup failed; continuing without tracing")
        try:
            await tracer.close()
        except Exception:
            logger.exception("Prefactor tracer cleanup failed after startup error")
        await session.start(agent=agent, **start_kwargs)
        return None


async def close_prefactor_tracer(tracer: PrefactorLiveKitSession | None) -> None:
    if tracer is None:
        return
    try:
        await tracer.close()
    except Exception:
        logger.exception("Prefactor tracer cleanup failed")


def get_runtime_clock_context() -> tuple[str, str]:
    now = datetime.now().astimezone()
    timezone_name = now.tzname() or "local time"
    timestamp = now.isoformat(timespec="seconds")
    return timestamp, timezone_name


def build_instructions(*, current_datetime: str, current_timezone: str) -> str:
    return f"""\
You are a voice-first web research assistant.

Runtime context:
- Current local datetime: {current_datetime}
- Current timezone: {current_timezone}

Behavior rules:
- If the user asks for facts, current events, comparisons, or anything that could
  plausibly be outdated, call search_web before answering.
- Search first, then synthesize. Do not pretend you searched if you did not.
- Start every research answer with a short spoken TLDR.
- After the TLDR, give 2 to 4 concise supporting points.
- End with a brief spoken source list that mentions the source title or domain.
- If sources conflict or evidence is weak, say that explicitly.
- Never invent sources, dates, or certainty.
- Keep answers compact and easy to follow in voice.
- Keep every reply under 200 words.
- You may answer simple greetings or conversational remarks without using search.

On your first message/turn please start off with ONLY the following:
    "Hi there, I am a web research agent that can search the web with up to date information. Please tell me what you'd like to search today!"
"""


class WebResearchAgent(Agent):
    def __init__(
        self,
        *,
        search_config: SearchConfig,
        exa_client: Exa,
        current_datetime: str,
        current_timezone: str,
    ) -> None:
        super().__init__(
            instructions=build_instructions(
                current_datetime=current_datetime,
                current_timezone=current_timezone,
            )
        )
        self._search_config = search_config
        self._exa_client = exa_client

    @function_tool()
    async def search_web(
        self,
        context: RunContext,
        query: str,
        num_results: int = DEFAULT_EXA_MAX_RESULTS,
        search_type: str = DEFAULT_EXA_SEARCH_TYPE,
        include_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search the web and return trimmed source material for synthesis."""

        del context
        return await asyncio.to_thread(
            run_exa_search,
            self._exa_client,
            self._search_config,
            query=query,
            num_results=num_results,
            search_type=search_type,
            include_domains=include_domains,
        )


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=DEFAULT_PREFACTOR_AGENT_ID)
async def web_research_agent(ctx: JobContext) -> None:
    config = resolve_agent_config()
    search_config = resolve_search_config()
    current_datetime, current_timezone = get_runtime_clock_context()
    ensure_turn_detector_assets()
    session = build_session(
        config,
        vad=ctx.proc.userdata.get("vad"),
        turn_detection=MultilingualModel(),
    )
    tracer = build_prefactor_tracer()
    agent = WebResearchAgent(
        search_config=search_config,
        exa_client=build_exa_client(search_config),
        current_datetime=current_datetime,
        current_timezone=current_timezone,
    )

    try:
        await start_session(
            session=session,
            agent=agent,
            tracer=tracer,
            room=ctx.room,
        )

        await ctx.connect()

        await session.generate_reply(
            instructions=(
                "Greet the user briefly. Tell them you can search the web and give a "
                "concise TLDR with sources. Mention that your current local datetime "
                f"is {current_datetime} in {current_timezone} if that is relevant. "
                "Keep the reply under 200 words. Ask what they want researched."
            )
        )
    except Exception:
        await close_prefactor_tracer(tracer)
        raise


def main() -> None:
    resolve_search_config()
    if should_validate_turn_detector():
        ensure_turn_detector_assets()
    cli.run_app(server)


if __name__ == "__main__":
    main()
