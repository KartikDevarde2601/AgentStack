from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.research.openai_client import OpenAIResearchClient


@pytest.mark.asyncio
async def test_openai_responses_json_object_requests_include_json_in_input() -> None:
    client = OpenAIResearchClient(api_key="sk-test", model="gpt-4.1-mini")
    captured: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> Any:
        captured.append(kwargs)
        if "sub-questions" in str(kwargs.get("input", "")):
            return SimpleNamespace(
                output_text=json.dumps({"sub_questions": ["Q1", "Q2", "Q3"]})
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
