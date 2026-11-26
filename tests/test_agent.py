from __future__ import annotations

import json
from unittest import mock

import pytest
from livekit.agents import AgentServer, AgentSession, inference, llm
from livekit.plugins import silero

import main


def test_server_is_registered() -> None:
    assert isinstance(main.server, AgentServer)
    assert main.server.setup_fnc is main.prewarm


def test_search_web_tool_schema_builds() -> None:
    agent = main.WebResearchAgent(search_config=main.resolve_search_config())
    (tool,) = agent.tools
    schema = llm.utils.build_strict_openai_schema(tool)
    assert schema["function"]["name"] == "search_web"
    assert set(schema["function"]["parameters"]["properties"]) == {
        "query",
        "num_results",
        "search_type",
        "include_domains",
    }


async def test_search_web_tool_calls_exa(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = main.WebResearchAgent(search_config=main.resolve_search_config())
    sentinel = {"ok": True}
    run = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(main, "run_exa_search", run)

    result = await agent.search_web(mock.Mock(), query="q", num_results=2)

    assert result is sentinel
    assert run.call_args.kwargs["query"] == "q"
    assert run.call_args.kwargs["num_results"] == 2


def test_silero_vad_loads() -> None:
    assert isinstance(silero.VAD.load(), silero.VAD)


@pytest.fixture
def turn_detector_runner():
    from livekit.plugins.turn_detector.multilingual import _EUORunnerMultilingual

    runner = _EUORunnerMultilingual()
    try:
        runner.initialize()
    except RuntimeError as exc:
        pytest.skip(f"turn-detector model not downloaded (run `mise run download-files`): {exc}")
    return runner


@pytest.mark.parametrize(
    ("text", "finished"),
    [
        ("What's the weather like in", False),
        ("What's the weather like in Paris today?", True),
    ],
)
def test_turn_detector_inference(turn_detector_runner, text: str, finished: bool) -> None:
    payload = json.dumps({"chat_ctx": [{"role": "user", "content": text}]}).encode()
    probability = json.loads(turn_detector_runner.run(payload))["eou_probability"]
    assert (probability > 0.5) is finished, probability


@pytest.mark.parametrize("preset_name", sorted(main.PRESETS))
async def test_agent_session_builds_like_entrypoint(
    livekit_env, monkeypatch: pytest.MonkeyPatch, preset_name: str
) -> None:
    """Build the AgentSession exactly as web_research_agent() does.

    The turn detector looks up the job's inference executor, which only exists
    inside a running worker, so that lookup is stubbed.
    """
    import livekit.plugins.turn_detector.base as turn_detector_base
    from livekit.plugins.turn_detector.multilingual import MultilingualModel

    monkeypatch.setattr(turn_detector_base, "get_job_context", mock.Mock())
    monkeypatch.setenv("AGENT_PRESET", preset_name)
    preset = main.resolve_preset()

    session = AgentSession(
        llm=inference.LLM(preset["llm_model"]),
        stt=inference.STT(model=preset["stt_model"], language=preset["stt_language"]),
        tts=inference.TTS(model=preset["tts_model"], voice=preset["tts_voice"]),
        vad=silero.VAD.load(),
        turn_handling={"turn_detection": MultilingualModel()},
        preemptive_generation=True,
    )
    assert isinstance(session, AgentSession)
