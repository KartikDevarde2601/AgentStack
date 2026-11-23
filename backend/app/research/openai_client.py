from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from openai import AsyncOpenAI

from app.domain import ResearchFinding, ResearchPlan


class ResearchLLMClient(Protocol):
    async def create_conversation(self) -> str: ...

    async def close_dangling_tool_calls(
        self, conversation_id: str, call_ids: list[str]
    ) -> None: ...

    async def generate_research_plan(
        self, query: str, *, conversation_id: str | None = None
    ) -> ResearchPlan: ...

    async def investigate_sub_question(
        self,
        query: str,
        sub_question: str,
        *,
        depth: str = "standard",
        conversation_id: str | None = None,
    ) -> ResearchFinding: ...

    async def synthesize_report_stream(
        self,
        query: str,
        findings: list[ResearchFinding],
        *,
        conversation_id: str | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str: ...


class OpenAIResearchClient:
    """Live OpenAI adapter using the Conversations and Responses APIs.

    Requires OPENAI_API_KEY at runtime (no silent offline fallback).
    """

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini") -> None:
        cleaned = api_key.strip()
        if not cleaned:
            raise RuntimeError(
                "OPENAI_API_KEY is required to run OpenAIResearchClient. "
                "Set OPENAI_API_KEY in your environment or .env file."
            )
        self._model = model
        self._client = AsyncOpenAI(api_key=cleaned)

    async def create_conversation(self) -> str:
        conversation = await self._client.conversations.create()
        return str(conversation.id)

    async def close_dangling_tool_calls(
        self, conversation_id: str, call_ids: list[str]
    ) -> None:
        if not conversation_id or not call_ids:
            return
        items: list[dict[str, Any]] = [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"status": "cancelled_by_user"}),
            }
            for call_id in call_ids
        ]
        await self._client.conversations.items.create(
            conversation_id=conversation_id,
            items=items,  # type: ignore[arg-type]
        )

    async def generate_research_plan(
        self, query: str, *, conversation_id: str | None = None
    ) -> ResearchPlan:
        cleaned_query = query.strip()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "instructions": (
                "You are a senior research planner. Break the user's question into "
                "3 distinct, focused sub-questions for parallel investigation. "
                'Return strict JSON with shape: {"sub_questions": ["...", "...", "..."]}.'
            ),
            "input": (
                f"Research query: {cleaned_query}\n\n"
                'Return the 3 sub-questions as a JSON object: {"sub_questions": [...]}.'
            ),
            "text": {"format": {"type": "json_object"}},
        }
        if conversation_id:
            kwargs["conversation"] = conversation_id

        response = await self._client.responses.create(**kwargs)
        raw_content = (getattr(response, "output_text", None) or "{}").strip()
        parsed = json.loads(raw_content)
        sub_questions = [
            str(item).strip()
            for item in parsed.get("sub_questions", [])
            if str(item).strip()
        ][:3]
        if not sub_questions:
            raise ValueError("OpenAI returned an empty research plan")
        return ResearchPlan(query=cleaned_query, sub_questions=sub_questions)

    async def investigate_sub_question(
        self,
        query: str,
        sub_question: str,
        *,
        depth: str = "standard",
        conversation_id: str | None = None,
    ) -> ResearchFinding:
        depth_guidance = {
            "concise": "Keep the summary brief (2-3 sentences) and 2 key points.",
            "standard": "Provide a balanced summary with 3 key points and concrete sources.",
            "deep": "Provide an in-depth technical analysis with 4-5 nuanced key points and sources.",
        }.get(depth, "Provide a balanced summary with 3 key points.")

        response = await self._client.responses.create(
            model=self._model,
            instructions=(
                "You are a focused research analyst. Investigate the sub-question in "
                f"the context of the main research query. {depth_guidance} "
                'Return strict JSON with shape: {"summary": "...", '
                '"key_points": ["...", "..."], "sources": ["..."]}.'
            ),
            input=(
                f"Main query: {query}\n"
                f"Sub-question: {sub_question}\n\n"
                'Return your finding as a JSON object: {"summary": "...", '
                '"key_points": [...], "sources": [...]}.'
            ),
            text={"format": {"type": "json_object"}},
        )
        raw_content = (getattr(response, "output_text", None) or "{}").strip()
        parsed = json.loads(raw_content)
        return ResearchFinding(
            sub_question=sub_question,
            summary=str(parsed.get("summary", "")).strip()
            or f"Key analysis regarding: {sub_question}",
            key_points=[
                str(point).strip()
                for point in parsed.get("key_points", [])
                if str(point).strip()
            ],
            sources=[
                str(src).strip()
                for src in parsed.get("sources", [])
                if str(src).strip()
            ],
        )

    async def synthesize_report_stream(
        self,
        query: str,
        findings: list[ResearchFinding],
        *,
        conversation_id: str | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        findings_payload = [finding.model_dump(mode="json") for finding in findings]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "instructions": (
                "You are a principal technical writer. Synthesize the provided "
                "research findings into a clear, well-structured Markdown report with: "
                "a title, Executive Summary, section per sub-question, Key Takeaways, "
                "and a Sources section."
            ),
            "input": (
                f"Main query: {query}\n\n"
                f"Findings JSON:\n{json.dumps(findings_payload, indent=2)}"
            ),
            "stream": True,
        }
        if conversation_id:
            kwargs["conversation"] = conversation_id

        chunks: list[str] = []
        stream = await self._client.responses.create(**kwargs)
        async for event in stream:
            if getattr(event, "type", None) == "response.output_text.delta":
                delta = str(getattr(event, "delta", "") or "")
                if delta:
                    chunks.append(delta)
                    if on_delta is not None:
                        await on_delta(delta)

        content = "".join(chunks).strip()
        if not content:
            raise ValueError("OpenAI returned an empty synthesized report")
        return content
