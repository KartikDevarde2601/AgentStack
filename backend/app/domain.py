from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


class ClarificationOption(BaseModel):
    """Selectable option for single_select or multi_select clarifying questions."""

    value: str
    label: str
    description: str | None = None


class ClarificationQuestion(BaseModel):
    """A clarifying question whose UI control is chosen dynamically by the agent."""

    id: str
    question: str
    kind: Literal["single_select", "multi_select", "short_text", "long_text"] = (
        "short_text"
    )
    options: list[ClarificationOption] = Field(default_factory=list)
    placeholder: str = ""
    required: bool = True


class RequestContext(BaseModel):
    user_id: str = "local-demo"


class ResearchPhase(StrEnum):
    IDLE = "idle"
    QUEUED = "queued"
    AWAITING_CLARIFICATIONS = "awaiting_clarifications"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (
            ResearchPhase.IDLE,
            ResearchPhase.COMPLETED,
            ResearchPhase.INTERRUPTED,
            ResearchPhase.FAILED,
        )


class ClarificationSubmission(BaseModel):
    turn_id: str | None = None
    skipped: bool = False
    answers: dict[str, str] = Field(default_factory=dict)


class PendingClarification(BaseModel):
    turn_id: str
    clarification_id: str
    intro: str = ""
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    widget_item_id: str


class ApprovalDecision(BaseModel):
    turn_id: str | None = None
    approved: bool
    sub_questions: list[str] = Field(default_factory=list)
    depth: Literal["concise", "standard", "deep"] = "standard"
    extra_focus: str = ""
    reason: str | None = None


class ResearchPlan(BaseModel):
    query: str
    sub_questions: list[str] = Field(default_factory=list)


class ConversationReply(BaseModel):
    kind: Literal["answer", "clarify"]
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


TurnResponse = ResearchPlan | ConversationReply


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
    enhanced_prompt: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    completed_sub_questions: int = 0
    total_sub_questions: int = 0
    findings: list[ResearchFinding] = Field(default_factory=list)
    approval_decision: ApprovalDecision | None = None
    report: str | None = None
    error: str | None = None


class IncomingMessage(BaseModel):
    message_id: str
    text: str
    already_persisted: bool = False


class PendingApproval(BaseModel):
    turn_id: str
    approval_id: str
    openai_call_id: str | None = None
    sub_questions: list[str] = Field(default_factory=list)
    widget_item_id: str


class ThreadWorkflowState(BaseModel):
    thread_id: str
    openai_conversation_id: str | None = None
    phase: ResearchPhase = ResearchPhase.IDLE
    active_turn_id: str | None = None
    message: str = ""
    pending_clarification: PendingClarification | None = None
    pending_approval: PendingApproval | None = None
    turns_since_continue: int = 0


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
