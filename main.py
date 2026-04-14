"""
Voice-first LiveKit web research agent with optional Prefactor tracing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os

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
from prefactor_core import SchemaRegistry
from prefactor_livekit import PrefactorLiveKitSession

load_dotenv()

DEFAULT_PRESET = "budget"
DEFAULT_EXA_MAX_RESULTS = 5
DEFAULT_EXA_SEARCH_TYPE = "auto"
MAX_EXA_RESULTS = 5
SEARCH_RESULT_TEXT_CHAR_LIMIT = 1600
DEFAULT_PREFACTOR_AGENT_ID = "web-research-agent"
DEFAULT_PREFACTOR_AGENT_NAME = "Web Research Agent"
VALID_EXA_SEARCH_TYPES = frozenset(
    {
        "auto",
        "deep",
        "deep-lite",
        "deep-max",
        "deep-reasoning",
        "fast",
        "hybrid",
        "instant",
        "keyword",
        "magic",
        "neural",
    }
)


# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchConfig:
    api_key: str
    max_results: int
    search_type: str
    include_domains: tuple[str, ...]


PRESETS = {
    "budget": {
        "llm_model": "openai/gpt-5-mini",
        "stt_model": "deepgram/flux-general",
        "stt_language": "en",
        "tts_model": "deepgram/aura-2",
        "tts_voice": "athena",
    },
    "balanced": {
        "llm_model": "openai/gpt-5.4",
        "stt_model": "deepgram/nova-3",
        "stt_language": "en",
        "tts_model": "cartesia/sonic-3",
        "tts_voice": "a4a16c5e-5902-4732-b9b6-2a48efd2e11b",
    },
}


def resolve_preset() -> dict[str, str]:
    preset_name = os.getenv("AGENT_PRESET", DEFAULT_PRESET).strip() or DEFAULT_PRESET
    return PRESETS.get(preset_name, PRESETS[DEFAULT_PRESET])


def resolve_search_config() -> SearchConfig:
    raw_max_results = (os.getenv("EXA_SEARCH_MAX_RESULTS") or "").strip()
    try:
        max_results = int(raw_max_results) if raw_max_results else DEFAULT_EXA_MAX_RESULTS
    except ValueError:
        max_results = DEFAULT_EXA_MAX_RESULTS

    search_type = (os.getenv("EXA_SEARCH_TYPE") or DEFAULT_EXA_SEARCH_TYPE).strip()
    if search_type not in VALID_EXA_SEARCH_TYPES:
        search_type = DEFAULT_EXA_SEARCH_TYPE

    return SearchConfig(
        api_key=(os.getenv("EXA_API_KEY") or "").strip(),
        max_results=min(max(1, max_results), MAX_EXA_RESULTS),
        search_type=search_type,
        include_domains=tuple(
            item.strip()
            for item in (os.getenv("EXA_INCLUDE_DOMAINS") or "").split(",")
            if item.strip()
        ),
    )


# ---------------------------------------------------------------------------
# Exa tool call
# ---------------------------------------------------------------------------


def _extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    return url.split("/")[2].lower() if "://" in url else None


def _trim_text(value: str | None, *, limit: int = SEARCH_RESULT_TEXT_CHAR_LIMIT) -> str:
    if not value:
        return ""
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _result_to_dict(result: Result) -> dict[str, str | None]:
    return {
        "title": result.title,
        "url": result.url,
        "domain": _extract_domain(result.url),
        "published_date": result.published_date,
        "author": result.author,
        "text": _trim_text(result.text),
    }


def run_exa_search(
    client: Exa,
    search_config: SearchConfig,
    *,
    query: str,
    num_results: int = DEFAULT_EXA_MAX_RESULTS,
    search_type: str = DEFAULT_EXA_SEARCH_TYPE,
    include_domains: list[str] | None = None,
) -> dict[str, object]:
    resolved_num_results = min(max(1, num_results), search_config.max_results)
    resolved_search_type = search_type.strip() or search_config.search_type
    if resolved_search_type not in VALID_EXA_SEARCH_TYPES:
        resolved_search_type = search_config.search_type

    resolved_domains = [item.strip() for item in include_domains or [] if item.strip()]
    if include_domains is None:
        resolved_domains = list(search_config.include_domains)

    response: SearchResponse[Result] = client.search(
        query,
        contents={"text": True},
        num_results=resolved_num_results,
        include_domains=resolved_domains or None,
        type=resolved_search_type,
    )
    results = [_result_to_dict(result) for result in response.results]

    return {
        "query": query,
        "resolved_search_type": response.resolved_search_type,
        "search_time_ms": response.search_time,
        "result_count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Prefactor
# ---------------------------------------------------------------------------


def build_prefactor_tool_schemas() -> dict[str, dict[str, object]]:
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


def build_schema_registry() -> SchemaRegistry:
    registry = SchemaRegistry()
    registry.register_type(
        name="example:session_setup",
        params_schema={
            "type": "object",
            "properties": {
                "preset": {"type": "string"},
                "search_provider": {"type": "string"},
                "tracing_enabled": {"type": "boolean"},
            },
            "required": ["preset", "search_provider", "tracing_enabled"],
        },
        result_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
            "required": ["status"],
        },
        title="Session Setup",
        description="Example app setup before the LiveKit session begins.",
    )
    return registry


def build_prefactor_tracer() -> PrefactorLiveKitSession | None:
    api_url = (os.getenv("PREFACTOR_API_URL") or "").strip()
    api_token = (os.getenv("PREFACTOR_API_TOKEN") or "").strip()
    if not api_url or not api_token:
        return None

    return PrefactorLiveKitSession.from_config(
        api_url=api_url,
        api_token=api_token,
        agent_id=(os.getenv("PREFACTOR_AGENT_ID") or DEFAULT_PREFACTOR_AGENT_ID),
        agent_name=(os.getenv("PREFACTOR_AGENT_NAME") or DEFAULT_PREFACTOR_AGENT_NAME),
        schema_registry=build_schema_registry(),
        tool_schemas=build_prefactor_tool_schemas(),
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


INSTRUCTIONS = """\
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
    def __init__(self, *, search_config: SearchConfig) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self._search_config = search_config
        self._exa_client = Exa(search_config.api_key)

    @function_tool()
    async def search_web(
        self,
        context: RunContext,
        query: str,
        num_results: int = DEFAULT_EXA_MAX_RESULTS,
        search_type: str = DEFAULT_EXA_SEARCH_TYPE,
        include_domains: list[str] | None = None,
    ) -> dict[str, object]:
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


# ---------------------------------------------------------------------------
# LiveKit app
# ---------------------------------------------------------------------------


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=DEFAULT_PREFACTOR_AGENT_ID)
async def web_research_agent(ctx: JobContext) -> None:
    preset_name = os.getenv("AGENT_PRESET", DEFAULT_PRESET).strip() or DEFAULT_PRESET
    preset = resolve_preset()
    search_config = resolve_search_config()
    session = AgentSession(
        llm=inference.LLM(preset["llm_model"]),
        stt=inference.STT(
            model=preset["stt_model"],
            language=preset["stt_language"],
        ),
        tts=inference.TTS(
            model=preset["tts_model"],
            voice=preset["tts_voice"],
        ),
        vad=ctx.proc.userdata.get("vad"),
        turn_handling={"turn_detection": MultilingualModel()},
        preemptive_generation=True,
    )
    agent = WebResearchAgent(search_config=search_config)
    tracer = build_prefactor_tracer()

    try:
        if tracer is None:
            await session.start(agent=agent, room=ctx.room)
        else:
            try:
                instance = await tracer.ensure_initialized()
                async with instance.span("example:session_setup") as span:
                    await span.start(
                        {
                            "preset": preset_name,
                            "search_provider": "exa",
                        }
                    )
                    await span.complete({"status": "ready"})
                await tracer.start(session=session, agent=agent, room=ctx.room)
            except Exception:
                try:
                    await tracer.close()
                except Exception:
                    pass
                await session.start(agent=agent, room=ctx.room)

        await ctx.connect()
        await session.generate_reply(
            instructions="Greet the user and ask what they want researched."
        )
    except Exception:
        if tracer is not None:
            try:
                await tracer.close()
            except Exception:
                pass
        raise


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
