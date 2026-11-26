from __future__ import annotations

import pytest

import main


def test_default_preset_is_budget() -> None:
    assert main.resolve_preset() == main.PRESETS["budget"]


def test_named_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PRESET", "balanced")
    assert main.resolve_preset() == main.PRESETS["balanced"]


def test_unknown_preset_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_PRESET", "nope")
    assert main.resolve_preset() == main.PRESETS[main.DEFAULT_PRESET]


def test_search_config_defaults() -> None:
    config = main.resolve_search_config()
    assert config == main.SearchConfig(
        api_key="",
        max_results=main.DEFAULT_EXA_MAX_RESULTS,
        search_type=main.DEFAULT_EXA_SEARCH_TYPE,
        include_domains=(),
    )


def test_search_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "  key  ")
    monkeypatch.setenv("EXA_SEARCH_MAX_RESULTS", "3")
    monkeypatch.setenv("EXA_SEARCH_TYPE", "fast")
    monkeypatch.setenv("EXA_INCLUDE_DOMAINS", " a.com, ,b.org ")
    config = main.resolve_search_config()
    assert config.api_key == "key"
    assert config.max_results == 3
    assert config.search_type == "fast"
    assert config.include_domains == ("a.com", "b.org")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 1), ("-4", 1), ("99", main.MAX_EXA_RESULTS), ("abc", main.DEFAULT_EXA_MAX_RESULTS)],
)
def test_search_config_clamps_max_results(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv("EXA_SEARCH_MAX_RESULTS", raw)
    assert main.resolve_search_config().max_results == expected


def test_search_config_rejects_unknown_search_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_SEARCH_TYPE", "bogus")
    assert main.resolve_search_config().search_type == main.DEFAULT_EXA_SEARCH_TYPE


@pytest.mark.parametrize("preset_name", sorted(main.PRESETS))
def test_preset_models_are_offered_by_livekit_inference(preset_name: str) -> None:
    """Catch a livekit-agents upgrade dropping a model id our presets use."""
    import typing

    from livekit.agents.inference import LLMModels, STTModels, TTSModels

    def literals(tp: object) -> set[str]:
        args = typing.get_args(tp)
        if not args:
            return {tp} if isinstance(tp, str) else set()
        return set().union(*(literals(a) for a in args))

    preset = main.PRESETS[preset_name]
    assert preset["llm_model"] in literals(LLMModels)
    assert preset["stt_model"] in literals(STTModels)
    assert preset["tts_model"] in literals(TTSModels)
