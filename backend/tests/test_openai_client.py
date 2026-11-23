from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.domain import ConversationReply, ResearchFinding, ResearchPlan
from app.research.openai_client import OpenAIResearchClient


@pytest.mark.asyncio
async def test_openai_responses_json_object_requests_include_json_in_input() -> None:
    client = OpenAIResearchClient(api_key="sk-test", model="gpt-4.1-mini")
    captured: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> Any:
        captured.append(kwargs)
        if "sub-questions" in str(kwargs.get("input", "")):
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "topic": "Compare Postgres and SQLite",
                        "sub_questions": ["Q1", "Q2", "Q3"],
                    }
                )
            )
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "summary": "Summary",
                    "key_points": ["P1"],
                    "sources": ["https://example.com"],
                }
            )
        )

    client._client.responses.create = fake_create  # type: ignore[method-assign]

    plan = await client.generate_research_plan(
        "Compare Postgres and SQLite", conversation_id="conv_123"
    )
    assert plan.sub_questions == ["Q1", "Q2", "Q3"]

    finding = await client.investigate_sub_question(
        "Compare Postgres and SQLite", "Q1", depth="deep"
    )
    assert finding.summary == "Summary"

    assert len(captured) == 2
    for req in captured:
        assert req["text"] == {"format": {"type": "json_object"}}
        assert "json" in str(req["input"]).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"kind": "answer", "text": "Write-ahead logging."},
            ConversationReply(kind="answer", text="Write-ahead logging."),
        ),
        (
            {"kind": "clarify", "text": "A quick answer or deep research?"},
            ConversationReply(kind="clarify", text="A quick answer or deep research?"),
        ),
        (
            {
                "kind": "research",
                "topic": "WAL tradeoffs",
                "sub_questions": ["Q1", "Q2", "Q3"],
            },
            ResearchPlan(query="WAL tradeoffs", sub_questions=["Q1", "Q2", "Q3"]),
        ),
    ],
)
async def test_resolve_turn_returns_the_selected_response(
    payload: dict[str, Any], expected: ConversationReply | ResearchPlan
) -> None:
    client = OpenAIResearchClient(api_key="sk-test")

    async def fake_create(**kwargs: Any) -> Any:
        assert kwargs["conversation"] == "conv_123"
        assert kwargs["input"] == "Can you go further?"
        assert kwargs["text"]["format"]["type"] == "json_schema"
        assert kwargs["text"]["format"]["strict"] is True
        return SimpleNamespace(output_text=json.dumps(payload))

    client._client.responses.create = fake_create  # type: ignore[method-assign]
    reply = await client.resolve_turn("Can you go further?", conversation_id="conv_123")
    assert reply == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_synthesis_streams_only_markdown(wrapped: bool) -> None:
    client = OpenAIResearchClient(api_key="sk-test")
    markdown = "# Report\n\n## Executive Summary\n\nThe findings."
    provider_text = json.dumps({"report": markdown}) if wrapped else markdown
    emitted: list[str] = []

    async def emit(delta: str) -> None:
        emitted.append(delta)

    async def provider_stream():
        yield SimpleNamespace(type="response.output_text.delta", delta=provider_text[:3])
        assert emitted == ([] if wrapped else [markdown[:3]])
        yield SimpleNamespace(type="response.output_text.delta", delta=provider_text[3:])

    async def fake_create(**kwargs: Any) -> Any:
        assert "conversation" not in kwargs
        assert kwargs["text"] == {"format": {"type": "text"}}
        return provider_stream()

    async def fake_append(conversation_id: str, messages: list[dict[str, str]]) -> None:
        assert conversation_id == "conv_123"
        assert messages == [{"role": "assistant", "content": markdown}]

    client._client.responses.create = fake_create  # type: ignore[method-assign]
    client.append_conversation_messages = fake_append  # type: ignore[method-assign]
    report = await client.synthesize_report_stream(
        "Compare Postgres and SQLite",
        [ResearchFinding(sub_question="How do they differ?", summary="Concurrency.")],
        conversation_id="conv_123",
        on_delta=emit,
    )
    assert report == markdown
    assert "".join(emitted) == markdown
