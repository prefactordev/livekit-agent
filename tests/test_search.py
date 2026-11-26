from __future__ import annotations

import inspect
from typing import Any

import jsonschema
from exa_py import Exa
from exa_py.api import Result, SearchResponse

import main

RESULT_SCHEMA = main.build_prefactor_tool_schemas()["search_web"]["result_schema"]
INPUT_SCHEMA = main.build_prefactor_tool_schemas()["search_web"]["input_schema"]


class FakeExa:
    """Stands in for exa_py.Exa, returning real exa_py response objects."""

    def __init__(self, results: list[Result] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._results = results if results is not None else [_result()]

    def search(self, query: str, **kwargs: Any) -> SearchResponse[Result]:
        self.calls.append((query, kwargs))
        return SearchResponse(
            results=self._results,
            resolved_search_type="neural",
            auto_date=None,
            search_time=123.0,
        )


def _result(**overrides: Any) -> Result:
    fields: dict[str, Any] = {
        "url": "https://Example.com/article",
        "id": "r1",
        "title": "A title",
        "published_date": "2026-01-01",
        "author": "Someone",
        "text": "Some   body\ntext",
    }
    fields.update(overrides)
    return Result(**fields)


def _config(**overrides: Any) -> main.SearchConfig:
    fields: dict[str, Any] = {
        "api_key": "k",
        "max_results": 5,
        "search_type": "auto",
        "include_domains": (),
    }
    fields.update(overrides)
    return main.SearchConfig(**fields)


def test_exa_search_accepts_the_arguments_we_pass() -> None:
    """Fail if an exa-py upgrade renames or drops a search() keyword we use."""
    params = inspect.signature(Exa.search).parameters
    for name in ("contents", "num_results", "include_domains", "type"):
        assert name in params, f"Exa.search no longer accepts {name!r}"


def test_run_exa_search_shapes_results() -> None:
    client = FakeExa()
    out = main.run_exa_search(client, _config(), query="q")  # type: ignore[arg-type]

    assert out == {
        "query": "q",
        "resolved_search_type": "neural",
        "search_time_ms": 123.0,
        "result_count": 1,
        "results": [
            {
                "title": "A title",
                "url": "https://Example.com/article",
                "domain": "example.com",
                "published_date": "2026-01-01",
                "author": "Someone",
                "text": "Some body text",
            }
        ],
    }
    jsonschema.validate(out, RESULT_SCHEMA)


def test_run_exa_search_request_arguments() -> None:
    client = FakeExa()
    main.run_exa_search(
        client,  # type: ignore[arg-type]
        _config(max_results=3),
        query="q",
        num_results=10,
        search_type="fast",
        include_domains=[" a.com ", ""],
    )
    assert client.calls == [
        (
            "q",
            {
                "contents": {"text": True},
                "num_results": 3,
                "include_domains": ["a.com"],
                "type": "fast",
            },
        )
    ]


def test_run_exa_search_uses_config_defaults() -> None:
    client = FakeExa()
    main.run_exa_search(
        client,  # type: ignore[arg-type]
        _config(search_type="neural", include_domains=("x.com",)),
        query="q",
        search_type="not-a-type",
    )
    kwargs = client.calls[0][1]
    assert kwargs["type"] == "neural"
    assert kwargs["include_domains"] == ["x.com"]


def test_run_exa_search_empty_domains_sent_as_none() -> None:
    client = FakeExa()
    main.run_exa_search(client, _config(), query="q", include_domains=[])  # type: ignore[arg-type]
    assert client.calls[0][1]["include_domains"] is None


def test_run_exa_search_handles_sparse_results() -> None:
    client = FakeExa([_result(url="", title=None, author=None, published_date=None, text=None)])
    out = main.run_exa_search(client, _config(), query="q")  # type: ignore[arg-type]
    assert out["results"][0]["domain"] is None
    assert out["results"][0]["text"] == ""
    jsonschema.validate(out, RESULT_SCHEMA)


def test_long_text_is_trimmed() -> None:
    client = FakeExa([_result(text="word " * 1000)])
    out = main.run_exa_search(client, _config(), query="q")  # type: ignore[arg-type]
    text = out["results"][0]["text"]
    assert len(text) == main.SEARCH_RESULT_TEXT_CHAR_LIMIT
    assert text.endswith("...")


def test_input_schema_accepts_tool_arguments() -> None:
    jsonschema.validate(
        {"query": "q", "num_results": 2, "search_type": "auto", "include_domains": None},
        INPUT_SCHEMA,
    )
