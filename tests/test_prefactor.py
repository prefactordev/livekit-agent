from __future__ import annotations

import inspect

import jsonschema
import pytest
from prefactor_core import SchemaRegistry
from prefactor_livekit import PrefactorLiveKitSession

import main


def test_tracer_disabled_without_config() -> None:
    assert main.build_prefactor_tracer() is None


def test_tracer_disabled_with_partial_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREFACTOR_API_URL", "https://prefactor.invalid")
    assert main.build_prefactor_tracer() is None


async def test_tracer_builds_with_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the tracer must not touch the network."""
    monkeypatch.setenv("PREFACTOR_API_URL", "https://prefactor.invalid")
    monkeypatch.setenv("PREFACTOR_API_TOKEN", "token")
    tracer = main.build_prefactor_tracer()
    assert isinstance(tracer, PrefactorLiveKitSession)
    await tracer.close()


def test_tracer_has_the_methods_main_calls() -> None:
    for name in ("ensure_initialized", "start", "close"):
        assert inspect.iscoroutinefunction(getattr(PrefactorLiveKitSession, name)), name


def test_schema_registry_builds() -> None:
    assert isinstance(main.build_schema_registry(), SchemaRegistry)


def _session_setup_params_schema() -> dict:
    captured: dict = {}

    class CapturingRegistry(SchemaRegistry):
        def register_type(self, name, params_schema, *args, **kwargs):  # type: ignore[override]
            captured[name] = params_schema
            return super().register_type(name, params_schema, *args, **kwargs)

    original = main.SchemaRegistry
    main.SchemaRegistry = CapturingRegistry  # type: ignore[misc]
    try:
        main.build_schema_registry()
    finally:
        main.SchemaRegistry = original  # type: ignore[misc]
    return captured["example:session_setup"]


@pytest.mark.parametrize("preset_name", sorted(main.PRESETS))
def test_session_setup_span_payload_matches_schema(preset_name: str) -> None:
    jsonschema.validate(main.session_setup_params(preset_name), _session_setup_params_schema())
