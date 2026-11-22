from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class RequestContext(BaseModel):
    user_id: str = "local-demo"


class ResearchPhase(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (ResearchPhase.COMPLETED, ResearchPhase.FAILED)


class ApprovalDecision(BaseModel):
    approved: bool
    sub_questions: list[str] = Field(default_factory=list)
    depth: Literal["concise", "standard", "deep"] = "standard"
    extra_focus: str = ""
    reason: str | None = None


class ResearchPlan(BaseModel):
    query: str
    sub_questions: list[str] = Field(default_factory=list)


class ResearchFinding(BaseModel):
    sub_question: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class ResearchProgress(BaseModel):
    workflow_id: str
    query: str
    phase: ResearchPhase
    message: str
    sub_questions: list[str] = Field(default_factory=list)
    completed_sub_questions: int = 0
    total_sub_questions: int = 0
    findings: list[ResearchFinding] = Field(default_factory=list)
    approval_decision: ApprovalDecision | None = None
    report: str | None = None
    error: str | None = None


class ThreadSummary(BaseModel):
    id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    item_count: int = 0
    latest_phase: ResearchPhase | None = None


class ResearchRunRecord(BaseModel):
    id: str
    thread_id: str
    workflow_id: str
    query: str
    phase: ResearchPhase
    message: str
    sub_questions: list[str] = Field(default_factory=list)
    report: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
