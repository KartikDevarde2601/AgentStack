from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing
from agents.tool import Tool
from temporalio.contrib.openai_agents.testing import ResponseBuilders

from app.research.agents.models import (
    ClarificationOption,
    ClarificationQuestion,
    Clarifications,
    EnhancedResearchBrief,
    ReportData,
    SearchSummary,
    WebSearchItem,
    WebSearchPlan,
)


def _flatten_input(input_val: str | list[TResponseInputItem]) -> str:
    if isinstance(input_val, str):
        return input_val
    parts: list[str] = []
    for item in input_val:
        if isinstance(item, dict):
            raw: dict[str, Any] = dict(item)
            content = raw.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        c_raw: dict[str, Any] = dict(c)
                        if c_raw.get("text"):
                            parts.append(str(c_raw["text"]))
            elif raw.get("output"):
                parts.append(str(raw["output"]))
    return "\n".join(parts)


class RoutingFakeModel(Model):
    """Deterministic OpenAI Agents SDK Model for Temporal integration tests.

    Routes responses based on the active agent's instructions / output schema,
    simulating real handoffs, dynamic clarifying questions, prompt enhancement,
    web search planning, parallel search summaries, and report synthesis.
    """

    def __init__(
        self,
        *,
        clarify_keywords: tuple[str, ...] = ("best", "broad", "melbourne"),
        direct_answers: dict[str, str] | None = None,
        fail_first_n_triage: int = 0,
        conversation_history: dict[str, list[str]] | None = None,
    ) -> None:
        self.clarify_keywords = clarify_keywords
        self.direct_answers = direct_answers or {}
        self._remaining_triage_failures = fail_first_n_triage
        self.conversation_history = (
            conversation_history if conversation_history is not None else {}
        )
        self.searched_queries: list[str] = []
        self.enhanced_prompts: list[str] = []
        self.triage_calls: list[tuple[str, str | None]] = []
        self._last_user_query: str = ""

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> ModelResponse:
        instructions = system_instructions or ""
        prompt_text = _flatten_input(input).strip()
        schema_name = output_schema.name() if output_schema is not None else ""

        # 1. Triage Agent (has handoffs, no output_schema)
        if "Triage Agent" in instructions:
            self.triage_calls.append((prompt_text, conversation_id))
            self._last_user_query = prompt_text
            if self._remaining_triage_failures > 0:
                self._remaining_triage_failures -= 1
                raise RuntimeError("400 Simulated upstream error")

            if prompt_text in self.direct_answers:
                return ResponseBuilders.output_message(self.direct_answers[prompt_text])

            lowered = prompt_text.lower()
            if any(kw in lowered for kw in self.clarify_keywords):
                return ResponseBuilders.tool_call(
                    arguments="{}",
                    name="transfer_to_clarifying_questions_agent",
                )
            return ResponseBuilders.tool_call(
                arguments="{}",
                name="transfer_to_research_instruction_agent",
            )

        # 2. Clarifying Questions Agent
        if schema_name == "Clarifications":
            clarifications = Clarifications(
                intro="A few quick details will help tailor this deep research.",
                questions=[
                    ClarificationQuestion(
                        id="q_0",
                        question="What is your primary goal?",
                        kind="single_select",
                        options=[
                            ClarificationOption(
                                value="architecture", label="Architecture tradeoffs"
                            ),
                            ClarificationOption(
                                value="production", label="Production readiness"
                            ),
                        ],
                    ),
                    ClarificationQuestion(
                        id="q_1",
                        question="Which dimensions should we compare?",
                        kind="multi_select",
                        options=[
                            ClarificationOption(value="latency", label="Latency"),
                            ClarificationOption(value="cost", label="Cost"),
                            ClarificationOption(value="ops", label="Operations"),
                        ],
                    ),
                    ClarificationQuestion(
                        id="q_2",
                        question="Any specific constraints or context?",
                        kind="long_text",
                        placeholder="e.g. team size, workload scale...",
                    ),
                ],
            )
            return ResponseBuilders.output_message(clarifications.model_dump_json())

        # 3. Research Instruction Agent (Prompt Enhancement)
        if schema_name == "EnhancedResearchBrief":
            self.enhanced_prompts.append(prompt_text)
            history = self.conversation_history.get(conversation_id or "", [])
            effective_query = (
                prompt_text.splitlines()[0].removeprefix("Original user query:").strip()
                if prompt_text.startswith("Original user query:")
                else self._last_user_query or prompt_text
            )
            combined = " ".join(history + [effective_query])
            topic = effective_query
            if "temporal" in combined.lower() and "try again" in effective_query.lower():
                topic = "Temporal durable execution for deep research agents"
            brief = EnhancedResearchBrief(
                topic=topic or "Deep Research Topic",
                instructions=f"Conduct comprehensive primary-source research on: {topic}. {prompt_text}",
            )
            return ResponseBuilders.output_message(brief.model_dump_json())

        # 4. Planner Agent
        if schema_name == "WebSearchPlan":
            topic_line = ""
            for line in prompt_text.splitlines():
                if line.startswith("Research topic:"):
                    topic_line = line.removeprefix("Research topic:").strip()
                    break
            topic = topic_line or "Research Topic"
            plan = WebSearchPlan(
                topic=topic,
                searches=[
                    WebSearchItem(
                        query=f"Architecture of {topic}",
                        reason="Understand core system architecture",
                    ),
                    WebSearchItem(
                        query=f"Production tradeoffs of {topic}",
                        reason="Evaluate real-world operational tradeoffs",
                    ),
                ],
            )
            return ResponseBuilders.output_message(plan.model_dump_json())

        # 5. Search Agent (with WebSearchTool)
        if schema_name == "SearchSummary":
            search_term = ""
            for line in prompt_text.splitlines():
                if line.startswith("Search term:"):
                    search_term = line.removeprefix("Search term:").strip()
                    break
            self.searched_queries.append(search_term)
            summary = SearchSummary(
                query=search_term,
                summary=f"Web evidence summary for {search_term}.",
                key_points=[f"Key web insight about {search_term}"],
                sources=["https://example.com/primary-source"],
            )
            return ResponseBuilders.output_message(summary.model_dump_json())

        # 6. Writer Agent
        if schema_name == "ReportData":
            topic = ""
            for line in prompt_text.splitlines():
                if line.startswith("Research topic:"):
                    topic = line.removeprefix("Research topic:").strip()
                    break
            report = ReportData(
                short_summary=f"Concise summary for {topic}.",
                markdown_report=(
                    f"# Deep Research Report: {topic}\n\n"
                    f"## Executive Summary\n\nSynthesized live web research findings for {topic}.\n\n"
                    f"## Sources\n\n- https://example.com/primary-source\n"
                ),
                follow_up_questions=[
                    f"Benchmark comparison for {topic}",
                    f"Migration guide for {topic}",
                ],
            )
            return ResponseBuilders.output_message(report.model_dump_json())

        return ResponseBuilders.output_message(
            json.dumps({"note": "fallback", "input": prompt_text})
        )

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError("Streaming not used in RoutingFakeModel")
