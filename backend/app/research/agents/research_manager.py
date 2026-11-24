from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from agents import RunConfig, Runner, custom_span, gen_trace_id, trace

    from app.domain import ResearchFinding
    from app.research.agents.clarifying_agent import new_clarifying_agent
    from app.research.agents.instruction_agent import new_instruction_agent
    from app.research.agents.models import (
        ClarificationQuestion,
        Clarifications,
        EnhancedResearchBrief,
        ReportData,
        SearchSummary,
        WebSearchItem,
        WebSearchPlan,
    )
    from app.research.agents.planner_agent import new_planner_agent
    from app.research.agents.search_agent import new_search_agent
    from app.research.agents.triage_agent import new_triage_agent
    from app.research.agents.writer_agent import new_writer_agent


@dataclass
class TriageOutcome:
    kind: Literal["answer", "clarify", "research"]
    direct_answer: str | None = None
    clarifications: Clarifications | None = None
    brief: EnhancedResearchBrief | None = None


class DeepResearchManager:
    """Multi-agent deep research orchestrator executing inside AgentThreadWorkflow.

    All Runner.run() calls are durably executed as Temporal model activities via
    OpenAIAgentsPlugin.
    """

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model
        self.run_config = RunConfig()
        self.triage_agent = new_triage_agent(model)
        self.clarifying_agent = new_clarifying_agent(model)
        self.instruction_agent = new_instruction_agent(model)
        self.planner_agent = new_planner_agent(model)
        self.search_agent = new_search_agent(model)
        self.writer_agent = new_writer_agent(model)

    async def triage(
        self, query: str, *, conversation_id: str | None = None
    ) -> TriageOutcome:
        trace_id = gen_trace_id()
        with trace("Deep Research Triage", trace_id=trace_id):
            result = await Runner.run(
                self.triage_agent,
                query.strip(),
                conversation_id=conversation_id,
                run_config=self.run_config,
            )
            output = result.final_output
            if isinstance(output, Clarifications):
                if output.questions:
                    return TriageOutcome(kind="clarify", clarifications=output)
                # Empty questions fallback: enhance directly.
                brief = await self.enhance_query(
                    query, [], {}, skipped=True, conversation_id=conversation_id
                )
                return TriageOutcome(kind="research", brief=brief)
            if isinstance(output, EnhancedResearchBrief):
                return TriageOutcome(kind="research", brief=output)
            text = str(output or "").strip()
            return TriageOutcome(
                kind="answer",
                direct_answer=text
                or "I'm ready to help—what would you like to explore?",
            )

    async def enhance_query(
        self,
        original_query: str,
        questions: list[ClarificationQuestion],
        answers: dict[str, str],
        *,
        skipped: bool = False,
        conversation_id: str | None = None,
    ) -> EnhancedResearchBrief:
        trace_id = gen_trace_id()
        with trace("Deep Research Prompt Enhancement", trace_id=trace_id):
            if skipped or not questions:
                prompt = (
                    f"Original user query: {original_query}\n\n"
                    "The user skipped clarifying questions. Treat unstated preferences "
                    "as open-ended and produce a detailed research brief."
                )
            else:
                lines = [
                    f"Original user query: {original_query}",
                    "",
                    "User clarification responses:",
                ]
                for idx, q in enumerate(questions):
                    ans = (
                        answers.get(q.id) or answers.get(f"q_{idx}") or "No preference"
                    )
                    lines.append(f"- {q.question}: {ans}")
                prompt = "\n".join(lines)

            result = await Runner.run(
                self.instruction_agent,
                prompt,
                conversation_id=conversation_id,
                run_config=self.run_config,
            )
            return result.final_output_as(EnhancedResearchBrief)

    async def plan_searches(self, brief: EnhancedResearchBrief) -> WebSearchPlan:
        trace_id = gen_trace_id()
        with trace("Deep Research Planning", trace_id=trace_id):
            prompt = (
                f"Research topic: {brief.topic}\n\n"
                f"Enhanced research instructions:\n{brief.instructions}"
            )
            result = await Runner.run(
                self.planner_agent,
                prompt,
                run_config=self.run_config,
            )
            plan = result.final_output_as(WebSearchPlan)
            if not plan.topic.strip():
                plan.topic = brief.topic
            return plan

    async def perform_searches(
        self,
        brief: EnhancedResearchBrief,
        searches: list[WebSearchItem],
        *,
        depth: str = "standard",
        on_search_complete: (
            Callable[[int, int, ResearchFinding | None], Awaitable[None]] | None
        ) = None,
    ) -> list[ResearchFinding]:
        trace_id = gen_trace_id()
        with (
            trace("Deep Research Web Searches", trace_id=trace_id),
            custom_span("Search the web"),
        ):
            total = len(searches)
            tasks = [
                asyncio.create_task(self._search_one(brief, item, depth=depth))
                for item in searches
            ]
            findings: list[ResearchFinding] = []
            for completed, task in enumerate(workflow.as_completed(tasks), start=1):
                finding = await task
                if finding is not None:
                    findings.append(finding)
                if on_search_complete is not None:
                    await on_search_complete(completed, total, finding)
            return findings

    async def _search_one(
        self,
        brief: EnhancedResearchBrief,
        item: WebSearchItem,
        *,
        depth: str = "standard",
    ) -> ResearchFinding | None:
        depth_guidance = {
            "concise": "Keep the summary brief (2-3 sentences) with 2-3 key points.",
            "standard": "Provide a balanced summary with 3-4 key points and cited URLs.",
            "deep": "Provide an in-depth technical synthesis with 4-5 nuanced key points and primary URLs.",
        }.get(depth, "Provide a balanced summary with 3-4 key points.")

        prompt = (
            f"Research topic: {brief.topic}\n"
            f"Search term: {item.query}\n"
            f"Reason for searching: {item.reason}\n"
            f"Depth guidance: {depth_guidance}"
        )
        try:
            result = await Runner.run(
                self.search_agent,
                prompt,
                run_config=self.run_config,
            )
            output: Any = result.final_output
            if isinstance(output, SearchSummary):
                return ResearchFinding(
                    sub_question=item.query,
                    summary=output.summary.strip() or f"Findings for: {item.query}",
                    key_points=[p.strip() for p in output.key_points if p.strip()],
                    sources=[s.strip() for s in output.sources if s.strip()],
                )
            text = str(output or "").strip()
            if not text:
                return None
            return ResearchFinding(
                sub_question=item.query,
                summary=text,
                key_points=[],
                sources=[],
            )
        except Exception as exc:  # noqa: BLE001
            workflow.logger.warning(
                "Web search failed for query '%s': %s", item.query, exc
            )
            return None

    async def write_report(
        self,
        brief: EnhancedResearchBrief,
        findings: list[ResearchFinding],
        *,
        depth: str = "standard",
    ) -> ReportData:
        trace_id = gen_trace_id()
        with trace("Deep Research Synthesis", trace_id=trace_id):
            findings_payload = [f.model_dump(mode="json") for f in findings]
            prompt = (
                f"Research topic: {brief.topic}\n"
                f"Requested depth: {depth}\n\n"
                f"Enhanced research instructions:\n{brief.instructions}\n\n"
                f"Web search findings:\n{json.dumps(findings_payload, indent=2)}"
            )
            result = await Runner.run(
                self.writer_agent,
                prompt,
                run_config=self.run_config,
            )
            return result.final_output_as(ReportData)
