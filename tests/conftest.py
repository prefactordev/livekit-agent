"""Shared fixtures.

These tests run offline: no LiveKit, Exa or Prefactor network calls. They exist
to catch dependency upgrades that break the APIs main.py relies on.
"""

from __future__ import annotations

import pytest

# main.py calls load_dotenv() on import, so a developer's real .env can leak in.
# Clear every variable the app reads so each test starts from a known state.
APP_ENV_VARS = (
    "AGENT_PRESET",
    "EXA_API_KEY",
    "EXA_SEARCH_MAX_RESULTS",
    "EXA_SEARCH_TYPE",
    "EXA_INCLUDE_DOMAINS",
    "PREFACTOR_API_URL",
    "PREFACTOR_API_TOKEN",
    "PREFACTOR_AGENT_ID",
    "PREFACTOR_AGENT_NAME",
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "LIVEKIT_REMOTE_EOT_URL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def livekit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dummy LiveKit credentials so inference clients can be constructed."""
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.invalid")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    # 32+ bytes so pyjwt doesn't warn about a short HMAC key.
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret-" + "x" * 32)
