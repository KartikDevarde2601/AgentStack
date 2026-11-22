from __future__ import annotations

from typing import Literal

from chatkit.widgets import DynamicWidgetRoot, WidgetTemplate
from pydantic import BaseModel, Field

from app.domain import ResearchFinding, ResearchPhase, ResearchProgress

APPROVE_PLAN_ACTION = "research.approve_plan"
REJECT_PLAN_ACTION = "research.reject_plan"
TOGGLE_FINDING_ACTION = "research.toggle_finding"
FOLLOWUP_ACTION = "research.followup"

# Official ChatKit WidgetTemplate architecture: load declarative .widget files
# (version 1.0 JSON + Jinja2 template + JSON Schema) via WidgetTemplate.from_file().
plan_approval_template = WidgetTemplate.from_file("research_plan_approval.widget")
progress_tracker_template = WidgetTemplate.from_file("research_progress_tracker.widget")
findings_explorer_template = WidgetTemplate.from_file("research_findings_explorer.widget")
followup_actions_template = WidgetTemplate.from_file("research_followup_actions.widget")


def approval_widget_item_id(workflow_id: str) -> str:
    return f"wgt_approval_{workflow_id}"


def progress_widget_item_id(workflow_id: str) -> str:
    return f"wgt_progress_{workflow_id}"


def findings_widget_item_id(workflow_id: str) -> str:
    return f"wgt_findings_{workflow_id}"


def actions_widget_item_id(workflow_id: str) -> str:
    return f"wgt_actions_{workflow_id}"


class PlanApprovalWidgetData(BaseModel):
    workflow_id: str
    sub_questions: list[str]
    state: Literal["pending", "approved", "cancelled"] = "pending"
    selected_depth: Literal["concise", "standard", "deep"] = "standard"
    extra_focus: str = ""
    approve_action_type: str = APPROVE_PLAN_ACTION
    reject_action_type: str = REJECT_PLAN_ACTION
    badge_label: str = "Approved"
    badge_color: str = "success"
    status_icon: str = "check-circle"
    status_text: str = ""


class ProgressStepData(BaseModel):
    label: str
    icon: str
    status: str
    color: str


class ProgressTrackerWidgetData(BaseModel):
    status_text: str
    status_icon: str
    phase_label: str
    phase_color: str
    message: str
    steps: list[ProgressStepData]
    completed_sub_questions: int = 0
    total_sub_questions: int = 0


class FindingRowData(BaseModel):
    sub_question: str
    summary: str
    sources_badge: str
    badge_color: str
    is_open: bool
    next_expanded: list[int] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    sources_text: str = ""


class FindingsExplorerWidgetData(BaseModel):
    workflow_id: str
    status_text: str
    toggle_action_type: str = TOGGLE_FINDING_ACTION
    rows: list[FindingRowData]


class FollowupActionsWidgetData(BaseModel):
    workflow_id: str
    query: str
    followup_action_type: str = FOLLOWUP_ACTION


def build_plan_approval_widget(
    *,
    workflow_id: str,
    sub_questions: list[str],
    state: Literal["pending", "approved", "cancelled"] = "pending",
    selected_depth: Literal["concise", "standard", "deep"] = "standard",
    extra_focus: str = "",
) -> DynamicWidgetRoot:
    """Widget 1: Human-in-the-Loop research plan approval form card."""
    badge_label = "Approved" if state == "approved" else "Cancelled"
    badge_color = "success" if state == "approved" else "danger"
    status_icon = "check-circle" if state == "approved" else "bug"
    status_text = (
        f"Plan approved ({selected_depth}) • {len(sub_questions)} sub-question(s)"
        if state == "approved"
        else "Plan cancelled by user"
    )
    data = PlanApprovalWidgetData(
        workflow_id=workflow_id,
        sub_questions=sub_questions,
        state=state,
        selected_depth=selected_depth,
        extra_focus=extra_focus.strip(),
        badge_label=badge_label,
        badge_color=badge_color,
        status_icon=status_icon,
        status_text=status_text,
    )
    return plan_approval_template.build(data)


def build_progress_tracker_widget(progress: ResearchProgress) -> DynamicWidgetRoot:
    """Widget 2: Live multi-phase research progress tracker card."""
    phase_badges: dict[ResearchPhase, tuple[str, str, str]] = {
        ResearchPhase.QUEUED: ("Queued", "secondary", "clock"),
        ResearchPhase.PLANNING: ("Planning", "info", "compass"),
        ResearchPhase.AWAITING_APPROVAL: ("Awaiting Approval", "warning", "compass"),
        ResearchPhase.RESEARCHING: ("Researching", "discovery", "search"),
        ResearchPhase.SYNTHESIZING: ("Synthesizing", "info", "sparkle"),
        ResearchPhase.COMPLETED: ("Completed", "success", "check-circle"),
        ResearchPhase.FAILED: ("Failed", "danger", "bug"),
    }
    phase_label, phase_color, status_icon = phase_badges[progress.phase]

    step_order = [
        (ResearchPhase.PLANNING, "Plan sub-questions"),
        (ResearchPhase.AWAITING_APPROVAL, "Human approval"),
        (ResearchPhase.RESEARCHING, "Parallel research"),
        (ResearchPhase.SYNTHESIZING, "Synthesize report"),
    ]
    phase_rank = {
        ResearchPhase.QUEUED: 0,
        ResearchPhase.PLANNING: 1,
        ResearchPhase.AWAITING_APPROVAL: 2,
        ResearchPhase.RESEARCHING: 3,
        ResearchPhase.SYNTHESIZING: 4,
        ResearchPhase.COMPLETED: 5,
        ResearchPhase.FAILED: 5,
    }
    current_rank = phase_rank[progress.phase]

    steps: list[ProgressStepData] = []
    for step_phase, step_label in step_order:
        r = phase_rank[step_phase]
        if current_rank > r:
            steps.append(
                ProgressStepData(
                    label=step_label,
                    icon="check-circle",
                    status="Done",
                    color="success",
                )
            )
        elif current_rank == r and progress.phase != ResearchPhase.FAILED:
            steps.append(
                ProgressStepData(
                    label=step_label,
                    icon="clock",
                    status="Active",
                    color="info",
                )
            )
        else:
            steps.append(
                ProgressStepData(
                    label=step_label,
                    icon="empty-circle",
                    status="Pending",
                    color="secondary",
                )
            )

    data = ProgressTrackerWidgetData(
        status_text=f"Workflow {progress.workflow_id[-8:]}",
        status_icon=status_icon,
        phase_label=phase_label,
        phase_color=phase_color,
        message=progress.message,
        steps=steps,
        completed_sub_questions=progress.completed_sub_questions,
        total_sub_questions=progress.total_sub_questions,
    )
    return progress_tracker_template.build(data)


def build_findings_explorer_widget(
    *,
    workflow_id: str,
    findings: list[ResearchFinding],
    expanded_indices: set[int] | None = None,
) -> DynamicWidgetRoot:
    """Widget 3: Interactive ListView of research findings with in-place sync_action toggle."""
    expanded = expanded_indices or set()
    rows: list[FindingRowData] = []

    for idx, finding in enumerate(findings):
        is_open = idx in expanded
        next_expanded = sorted((expanded - {idx}) if is_open else (expanded | {idx}))
        rows.append(
            FindingRowData(
                sub_question=finding.sub_question,
                summary=finding.summary,
                sources_badge=f"{len(finding.sources)} source(s)",
                badge_color="info" if is_open else "secondary",
                is_open=is_open,
                next_expanded=next_expanded,
                key_points=list(finding.key_points),
                sources_text=(
                    "Sources: " + " • ".join(finding.sources)
                    if finding.sources
                    else ""
                ),
            )
        )

    data = FindingsExplorerWidgetData(
        workflow_id=workflow_id,
        status_text=f"Research Findings ({len(findings)}) • Click any row to expand",
        rows=rows,
    )
    return findings_explorer_template.build(data)


def build_followup_actions_widget(
    *,
    workflow_id: str,
    query: str,
) -> DynamicWidgetRoot:
    """Widget 4: Interactive post-report follow-up action buttons card."""
    data = FollowupActionsWidgetData(
        workflow_id=workflow_id,
        query=query,
    )
    return followup_actions_template.build(data)
