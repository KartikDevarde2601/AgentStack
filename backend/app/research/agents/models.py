from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain import ClarificationOption, ClarificationQuestion

__all__ = [
    "ClarificationOption",
    "ClarificationQuestion",
    "Clarifications",
    "EnhancedResearchBrief",
    "ReportData",
    "SearchSummary",
    "WebSearchItem",
    "WebSearchPlan",
]


class Clarifications(BaseModel):
    """Structured output returned by the Clarifying Questions Agent."""

    intro: str = ""
    questions: list[ClarificationQuestion] = Field(default_factory=list)


class EnhancedResearchBrief(BaseModel):
    """Prompt-enhanced research instructions produced by the Research Instruction Agent."""

    topic: str
    instructions: str


class WebSearchItem(BaseModel):
    """A single planned web search with justification."""

    reason: str
    query: str


class WebSearchPlan(BaseModel):
    """Set of targeted web searches planned by the Planner Agent."""

    topic: str
    searches: list[WebSearchItem] = Field(default_factory=list)


class SearchSummary(BaseModel):
    """Structured evidence returned by the Search Agent after WebSearchTool execution."""

    query: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class ReportData(BaseModel):
    """Final synthesized research report produced by the Writer Agent."""

    short_summary: str
    markdown_report: str
    follow_up_questions: list[str] = Field(default_factory=list)
