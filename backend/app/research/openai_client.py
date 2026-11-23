from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from openai import AsyncOpenAI

from app.domain import ConversationReply, ResearchFinding, ResearchPlan, TurnResponse


class ResearchLLMClient(Protocol):
    async def create_conversation(self) -> str: ...

    async def append_conversation_messages(
        self, conversation_id: str, messages: list[dict[str, str]]
    ) -> None: ...

    async def generate_research_plan(
        self, query: str, *, conversation_id: str | None = None
    ) -> ResearchPlan: ...

    async def resolve_turn(
        self, query: str, *, conversation_id: str | None = None
    ) -> TurnResponse: ...

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

    async def append_conversation_messages(
        self, conversation_id: str, messages: list[dict[str, str]]
    ) -> None:
        if not conversation_id or not messages:
            return
        items: list[dict[str, Any]] = [
            {
                "type": "message",
                "role": msg["role"],
                "content": msg["content"],
            }
            for msg in messages
            if msg.get("content")
        ]
        if not items:
            return
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
                "You are a senior research planner. Use the full conversation history "
                "to understand follow-ups, refinements, and retries (such as 'try again'). "
                "Identify the concrete standalone research topic and break it into "
                "3 distinct, focused sub-questions for parallel investigation. "
                'Return strict JSON with shape: {"topic": "...", "sub_questions": ["...", "...", "..."]}.'
            ),
            "input": (
                f"User message: {cleaned_query}\n\n"
                'Return the resolved research topic and 3 sub-questions as a JSON object: '
                '{"topic": "...", "sub_questions": [...]}.'
            ),
            "text": {"format": {"type": "json_object"}},
        }
        if conversation_id:
            kwargs["conversation"] = conversation_id

        response = await self._client.responses.create(**kwargs)
        raw_content = (getattr(response, "output_text", None) or "{}").strip()
        parsed = json.loads(raw_content)
        resolved_topic = str(parsed.get("topic") or "").strip() or cleaned_query
        sub_questions = [
            str(item).strip()
            for item in parsed.get("sub_questions", [])
            if str(item).strip()
        ][:3]
        if not sub_questions:
            raise ValueError("OpenAI returned an empty research plan")
        return ResearchPlan(query=resolved_topic, sub_questions=sub_questions)

    async def resolve_turn(
        self, query: str, *, conversation_id: str | None = None
    ) -> TurnResponse:
        response = await self._client.responses.create(
            model=self._model,
            conversation=conversation_id,
            instructions=(
                "You are AgentStack's conversational research assistant. Read the latest "
                "user message in the context of the conversation. Use earlier plans, reports, "
                "and replies to resolve references and interpret follow-ups or retries. "
                "Choose exactly one response and return it as JSON.\n"
                '1. Use kind "answer" for ordinary questions, explanations, '
                "or edits and summaries you can provide using the conversation or general "
                "knowledge. Put the reply the user should see in text.\n"
                '2. Use kind "research" '
                "when the user explicitly requests deep research or clearly needs a new "
                "multi-part investigation. Resolve the topic as a standalone request and "
                "decompose it into three distinct questions for parallel investigation. "
                "A short message can request research if the intent is clear.\n"
                '3. Use kind "clarify" when the history does not resolve '
                "what to do. Put one focused question in text. If the intended depth is unclear, "
                "ask whether the user wants a quick answer or deep research. If the topic "
                "is unclear, ask for the missing detail. Set fields you don't use to null."
            ),
            input=query.strip(),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "turn_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": ["answer", "research", "clarify"]},
                            "text": {"type": ["string", "null"]},
                            "topic": {"type": ["string", "null"]},
                            "sub_questions": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["kind", "text", "topic", "sub_questions"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        result = json.loads(response.output_text)
        if result.get("kind") != "research":
            return ConversationReply.model_validate(result)

        plan = ResearchPlan(query=result["topic"], sub_questions=result["sub_questions"])
        plan.query = plan.query.strip()
        plan.sub_questions = [question.strip() for question in plan.sub_questions]
        if not plan.query or len(plan.sub_questions) != 3 or not all(plan.sub_questions):
            raise ValueError("OpenAI returned an incomplete research plan")
        return plan

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
                "and a Sources section. Start with a Markdown title heading. Output only "
                "the Markdown report, with no JSON or enclosing code fence."
            ),
            "input": (
                f"Main query: {query}\n\n"
                f"Research findings:\n{json.dumps(findings_payload, indent=2)}"
            ),
            "text": {"format": {"type": "text"}},
            "stream": True,
        }
        # The resolved query and findings carry the report context. Generate outside
        # the conversation so earlier routing instructions cannot dictate its format.
        chunks: list[str] = []
        stream_markdown: bool | None = None
        stream = await self._client.responses.create(**kwargs)
        async for event in stream:
            if getattr(event, "type", None) == "response.output_text.delta":
                delta = str(getattr(event, "delta", "") or "")
                if delta:
                    chunks.append(delta)
                    if stream_markdown is None:
                        prefix = "".join(chunks).lstrip()
                        if prefix:
                            # Buffer unexpected formats so a wrapper cannot reach ChatKit.
                            stream_markdown = prefix.startswith("#")
                            if stream_markdown and on_delta is not None:
                                await on_delta("".join(chunks))
                    elif stream_markdown and on_delta is not None:
                        await on_delta(delta)

        content = "".join(chunks).strip()
        if not stream_markdown:
            candidate = content
            if candidate.startswith("```json\n") and candidate.endswith("```"):
                candidate = candidate.removeprefix("```json\n").removesuffix("```").strip()
            if candidate.startswith("{"):
                envelope = json.loads(candidate)
                report = envelope.get("report")
                if not isinstance(report, str):
                    raise ValueError("OpenAI returned JSON without a research report")
                content = report.strip()
            if content and on_delta is not None:
                await on_delta(content)
        if not content:
            raise ValueError("OpenAI returned an empty synthesized report")
        if conversation_id:
            await self.append_conversation_messages(
                conversation_id, [{"role": "assistant", "content": content}]
            )
        return content
