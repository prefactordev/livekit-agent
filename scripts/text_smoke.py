from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from livekit.agents import AgentSession, inference
from prefactor_http.exceptions import PrefactorValidationError

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def load_main_module() -> Any:
    """Load the vendored example `main.py` from disk."""
    module_path = EXAMPLE_ROOT / "main.py"
    spec = importlib.util.spec_from_file_location(
        "livekit_agent_example_main",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load vendored LiveKit example main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


main = load_main_module()


DEFAULT_QUERIES = [
    "What changed in Python 3.12 and give me the TLDR.",
    "Summarize the latest SpaceX Starship news in a few points.",
    "Compare Exa and Tavily for web research agents.",
]
DEFAULT_REQUIRED_SPAN_TYPES = (
    "livekit:session",
    "livekit:user_turn",
    "livekit:assistant_turn",
    "livekit:llm",
    "livekit:tool:search_web",
)


class PrefactorVerificationError(RuntimeError):
    pass


def extract_text_content(item: Any) -> str:
    content = getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if isinstance(part, str):
                fragments.append(part)
                continue
            if isinstance(part, dict) and part.get("type") == "text":
                fragments.append(str(part.get("text", "")))
            elif hasattr(part, "text"):
                fragments.append(str(part.text))
        return " ".join(fragment for fragment in fragments if fragment).strip()
    return ""


def extract_assistant_text(events: list[Any]) -> str | None:
    for event in reversed(events):
        if getattr(event, "type", None) != "message":
            continue
        item = getattr(event, "item", None)
        if getattr(item, "role", None) == "assistant":
            return extract_text_content(item)
    return None


def build_prefactor_window(started_at: float) -> tuple[str, str]:
    start_time = datetime.fromtimestamp(started_at, tz=UTC) - timedelta(seconds=30)
    end_time = datetime.now(tz=UTC) + timedelta(seconds=30)
    return (
        start_time.isoformat().replace("+00:00", "Z"),
        end_time.isoformat().replace("+00:00", "Z"),
    )


def build_prefactor_spans_url(instance_id: str, started_at: float) -> tuple[str, str]:
    start_time, end_time = build_prefactor_window(started_at)
    query_string = urllib.parse.urlencode(
        {
            "agent_instance_id": instance_id,
            "start_time": start_time,
            "end_time": end_time,
            "include_summaries": "true",
        }
    )
    return f"/api/v1/agent_spans?{query_string}", end_time


def _collect_span_records(payload: Any, results: list[dict[str, Any]]) -> None:
    if isinstance(payload, dict):
        if isinstance(payload.get("schema_name"), str):
            results.append(payload)
        for value in payload.values():
            _collect_span_records(value, results)
        return
    if isinstance(payload, list):
        for item in payload:
            _collect_span_records(item, results)


def extract_span_records(payload: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    _collect_span_records(payload, results)
    deduped: dict[str, dict[str, Any]] = {}
    for record in results:
        record_id = str(
            record.get("id") or f"{record.get('schema_name')}:{len(deduped)}"
        )
        deduped[record_id] = record
    return list(deduped.values())


def fetch_prefactor_json(path: str) -> Any:
    api_url = (os.environ.get("PREFACTOR_API_URL") or "").strip()
    api_token = (os.environ.get("PREFACTOR_API_TOKEN") or "").strip()
    if not api_url or not api_token:
        raise PrefactorVerificationError(
            "Prefactor verification requires PREFACTOR_API_URL and PREFACTOR_API_TOKEN."
        )

    request = urllib.request.Request(
        f"{api_url.rstrip('/')}{path}",
        headers={
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


async def verify_prefactor_spans(
    *,
    instance_id: str,
    started_at: float,
    required_span_types: tuple[str, ...],
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    latest_payload: Any = None
    latest_records: list[dict[str, Any]] = []

    while True:
        path, query_end_time = build_prefactor_spans_url(instance_id, started_at)
        latest_payload = await asyncio.to_thread(fetch_prefactor_json, path)
        latest_records = extract_span_records(latest_payload)
        found_span_types = sorted(
            {
                str(record["schema_name"])
                for record in latest_records
                if isinstance(record.get("schema_name"), str)
            }
        )
        missing_span_types = [
            span_type
            for span_type in required_span_types
            if span_type not in found_span_types
        ]
        if not missing_span_types:
            return {
                "ok": True,
                "instance_id": instance_id,
                "span_count": len(latest_records),
                "found_span_types": found_span_types,
                "missing_span_types": [],
                "query_end_time": query_end_time,
            }

        if time.monotonic() >= deadline:
            return {
                "ok": False,
                "instance_id": instance_id,
                "span_count": len(latest_records),
                "found_span_types": found_span_types,
                "missing_span_types": missing_span_types,
                "query_end_time": query_end_time,
                "raw_response": latest_payload,
            }

        await asyncio.sleep(poll_interval_seconds)


async def start_text_session(
    *,
    session: main.AgentSession,
    agent: main.Agent,
    tracer: main.PrefactorLiveKitSession | None,
    require_prefactor: bool,
) -> str | None:
    if tracer is None:
        if require_prefactor:
            raise PrefactorVerificationError(
                "Prefactor verification was requested, but tracing is disabled."
            )
        await session.start(agent=agent, record=False)
        return None

    if require_prefactor:
        try:
            instance = await tracer.ensure_initialized()
        except PrefactorValidationError as exc:
            error_details = json.dumps(exc.errors, indent=2, sort_keys=True)
            raise PrefactorVerificationError(
                "Prefactor tracing startup failed before the session began. "
                "This usually means PREFACTOR_AGENT_ID does not reference a real "
                f"Prefactor agent.\n{error_details}"
            ) from exc

        await tracer.start(session=session, agent=agent, record=False)
        return instance.id

    await main.start_session(
        session=session,
        agent=agent,
        tracer=tracer,
        record=False,
    )
    instance = getattr(tracer, "_instance", None)
    return getattr(instance, "id", None)


async def run_queries(
    queries: list[str],
    *,
    verify_prefactor: bool,
    required_span_types: tuple[str, ...],
    prefactor_timeout_seconds: float,
    prefactor_poll_interval_seconds: float,
) -> dict[str, Any]:
    config = main.resolve_agent_config()
    search_config = main.resolve_search_config()
    # Text-only smoke runs should not initialize the full STT/TTS stack because
    # the inference plugins expect a LiveKit worker/job HTTP context.
    session = AgentSession(
        llm=inference.LLM(config.llm_model),
        preemptive_generation=True,
    )
    agent = main.WebResearchAgent(
        search_config=search_config,
        exa_client=main.build_exa_client(search_config),
    )
    tracer = main.build_prefactor_tracer()
    started_at = time.time()
    instance_id: str | None = None
    results: list[dict[str, Any]] = []

    try:
        instance_id = await start_text_session(
            session=session,
            agent=agent,
            tracer=tracer,
            require_prefactor=verify_prefactor,
        )

        for query in queries:
            run_result = session.run(user_input=query)
            await run_result
            assistant_text = extract_assistant_text(run_result.events)
            results.append(
                {
                    "query": query,
                    "assistant_text": assistant_text,
                    "event_count": len(run_result.events),
                }
            )

        await session.aclose()
    finally:
        await main.close_prefactor_tracer(tracer)

    payload = {
        "instance_id": instance_id,
        "started_at": started_at,
        "queries": results,
    }
    if verify_prefactor:
        if not instance_id:
            raise PrefactorVerificationError(
                "Prefactor verification was requested, but no agent "
                "instance ID was created."
            )
        payload["prefactor_verification"] = await verify_prefactor_spans(
            instance_id=instance_id,
            started_at=started_at,
            required_span_types=required_span_types,
            timeout_seconds=prefactor_timeout_seconds,
            poll_interval_seconds=prefactor_poll_interval_seconds,
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text-only smoke test runner")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to run. May be provided multiple times.",
    )
    parser.add_argument(
        "--verify-prefactor",
        action="store_true",
        help="Fail if Prefactor spans are not emitted for the run.",
    )
    parser.add_argument(
        "--require-span",
        action="append",
        dest="required_span_types",
        help=(
            "Prefactor span type that must be present. May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--prefactor-timeout-seconds",
        type=float,
        default=20.0,
        help="How long to wait for Prefactor spans to appear.",
    )
    parser.add_argument(
        "--prefactor-poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval while waiting for Prefactor spans.",
    )
    return parser


def print_prefactor_hints(instance_id: str, started_at: float) -> None:
    start_time, end_time = build_prefactor_window(started_at)
    print("\nPrefactor CLI hints:")
    print(f"prefactor agent_instances retrieve {instance_id}")
    print(
        "prefactor agent_spans list "
        f"--agent_instance_id {instance_id} "
        f"--start_time {start_time} "
        f"--end_time {end_time} "
        "--include_summaries"
    )


async def async_main() -> None:
    args = build_parser().parse_args()
    queries = args.queries or DEFAULT_QUERIES
    required_span_types = tuple(args.required_span_types or DEFAULT_REQUIRED_SPAN_TYPES)
    payload = await run_queries(
        queries,
        verify_prefactor=args.verify_prefactor,
        required_span_types=required_span_types,
        prefactor_timeout_seconds=args.prefactor_timeout_seconds,
        prefactor_poll_interval_seconds=args.prefactor_poll_interval_seconds,
    )
    print(json.dumps(payload, indent=2))
    verification = payload.get("prefactor_verification")
    if verification and not verification.get("ok"):
        raise SystemExit(
            "Prefactor verification failed. Missing span types: "
            + ", ".join(verification["missing_span_types"])
        )
    if payload["instance_id"]:
        print_prefactor_hints(payload["instance_id"], payload["started_at"])


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except PrefactorVerificationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
