from __future__ import annotations

from pydantic import BaseModel
from temporalio import activity

from app.domain import ResearchFinding, ResearchPlan
from app.research.openai_client import OpenAIResearchClient


class GatherFindingInput(BaseModel):
    query: str
    sub_question: str


class SynthesizeReportInput(BaseModel):
    query: str
    findings: list[ResearchFinding]


class ResearchActivities:
    def __init__(self, client: OpenAIResearchClient) -> None:
        self._client = client

    @activity.defn(name="plan_research_activity")
    async def plan_research_activity(self, query: str) -> ResearchPlan:
        return await self._client.generate_research_plan(query)

    @activity.defn(name="gather_finding_activity")
    async def gather_finding_activity(
        self, params: GatherFindingInput
    ) -> ResearchFinding:
        return await self._client.investigate_sub_question(
            params.query, params.sub_question
        )

    @activity.defn(name="synthesize_report_activity")
    async def synthesize_report_activity(
        self, params: SynthesizeReportInput
    ) -> str:
        return await self._client.synthesize_report(params.query, params.findings)
