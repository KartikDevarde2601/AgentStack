from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from app.domain import ResearchFinding, ResearchPlan


class OpenAIResearchClient:
    """Boundary adapter wrapping openai.AsyncOpenAI with deterministic demo fallback."""

    def __init__(self, api_key: str = "", model: str = "gpt-4.1-mini") -> None:
        self._api_key = api_key.strip()
        self._model = model
        self._client = AsyncOpenAI(api_key=self._api_key) if self._api_key else None

    @property
    def is_live(self) -> bool:
        return self._client is not None

    async def generate_research_plan(self, query: str) -> ResearchPlan:
        cleaned_query = query.strip()
        if not self._client:
            return _demo_research_plan(cleaned_query)

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior research planner. Break the user's question into "
                        "exactly 3 distinct, focused sub-questions for parallel investigation. "
                        'Return JSON with shape: {"sub_questions": ["...", "...", "..."]}.'
                    ),
                },
                {"role": "user", "content": cleaned_query},
            ],
        )
        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)
        sub_questions = [
            str(item).strip()
            for item in parsed.get("sub_questions", [])
            if str(item).strip()
        ][:3]
        if not sub_questions:
            return _demo_research_plan(cleaned_query)
        return ResearchPlan(query=cleaned_query, sub_questions=sub_questions)

    async def investigate_sub_question(
        self, query: str, sub_question: str
    ) -> ResearchFinding:
        if not self._client:
            return _demo_finding(query, sub_question)

        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a focused research analyst. Investigate the sub-question in "
                        "the context of the main research query. Return JSON with shape: "
                        '{"summary": "...", "key_points": ["...", "..."], "sources": ["..."]}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Main query: {query}\nSub-question: {sub_question}",
                },
            ],
        )
        raw_content = response.choices[0].message.content or "{}"
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

    async def synthesize_report(
        self, query: str, findings: list[ResearchFinding]
    ) -> str:
        if not self._client:
            return _demo_synthesize_report(query, findings)

        findings_payload = [finding.model_dump(mode="json") for finding in findings]
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a principal technical writer. Synthesize the provided "
                        "research findings into a clear, well-structured Markdown report with: "
                        "a title, Executive Summary, section per sub-question, Key Takeaways, "
                        "and a Sources section."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Main query: {query}\n\n"
                        f"Findings JSON:\n{json.dumps(findings_payload, indent=2)}"
                    ),
                },
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        return content or _demo_synthesize_report(query, findings)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or "research-topic"


def _demo_research_plan(query: str) -> ResearchPlan:
    return ResearchPlan(
        query=query,
        sub_questions=[
            f"What are the foundational concepts and current landscape of {query}?",
            f"What are the primary architectural tradeoffs, benchmarks, and risks in {query}?",
            f"What practical best practices and near-term trends apply to {query}?",
        ],
    )


def _demo_finding(query: str, sub_question: str) -> ResearchFinding:
    slug = _slugify(query)
    sub_slug = _slugify(sub_question)
    return ResearchFinding(
        sub_question=sub_question,
        summary=(
            f"Analysis of '{sub_question}' highlights clear design drivers, "
            f"measurable operational tradeoffs, and repeatable patterns relevant to {query}."
        ),
        key_points=[
            f"Core framing: establishes the baseline criteria needed to evaluate {sub_question.lower()}",
            "Operational impact: balances reliability, latency, and maintainability under real workloads.",
            "Implementation guidance: favors observable workflows, typed boundaries, and incremental rollout.",
        ],
        sources=[
            f"https://docs.example.com/{slug}/overview",
            f"https://research.example.com/{slug}/{sub_slug}",
        ],
    )


def _demo_synthesize_report(query: str, findings: list[ResearchFinding]) -> str:
    lines: list[str] = [
        f"# Deep Research Report: {query}",
        "",
        "## Executive Summary",
        "",
        (
            f"This report synthesizes {len(findings)} parallel research tracks investigating "
            f"**{query}**. Together, the findings outline the current landscape, core tradeoffs, "
            "and concrete recommendations for practical execution."
        ),
        "",
        "## Detailed Findings",
        "",
    ]

    all_sources: list[str] = []
    for index, finding in enumerate(findings, start=1):
        lines.append(f"### {index}. {finding.sub_question}")
        lines.append("")
        lines.append(finding.summary)
        lines.append("")
        for point in finding.key_points:
            lines.append(f"- {point}")
        lines.append("")
        for src in finding.sources:
            if src not in all_sources:
                all_sources.append(src)

    lines.extend(
        [
            "## Recommendations",
            "",
            "- Start with a clear evaluation rubric tied to your concrete constraints.",
            "- Isolate external dependencies behind typed boundaries for safer iteration.",
            "- Instrument durable execution and progress tracking early to simplify debugging.",
            "",
            "## Sources",
            "",
        ]
    )
    for idx, src in enumerate(all_sources, start=1):
        lines.append(f"{idx}. [{src}]({src})")

    return "\n".join(lines)
