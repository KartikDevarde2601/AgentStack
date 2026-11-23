from __future__ import annotations

from pathlib import Path

from chatkit.widgets import WidgetTemplate

from app.domain import ResearchFinding, ResearchPhase, ResearchProgress
from app.research.agents.models import ClarificationOption, ClarificationQuestion
from app.widgets import (
    build_clarifications_widget,
    build_findings_explorer_widget,
    build_followup_actions_widget,
    build_plan_approval_widget,
    build_progress_tracker_widget,
    clarifications_template,
    findings_explorer_template,
    followup_actions_template,
    plan_approval_template,
    progress_tracker_template,
)


def test_all_five_widget_templates_are_loaded_from_dot_widget_files() -> None:
    widgets_dir = Path(__file__).resolve().parents[1] / "app" / "widgets"
    expected_files = {
        "research_clarifications.widget",
        "research_plan_approval.widget",
        "research_progress_tracker.widget",
        "research_findings_explorer.widget",
        "research_followup_actions.widget",
    }
    actual_files = {p.name for p in widgets_dir.glob("*.widget")}
    assert expected_files == actual_files

    for tpl in (
        clarifications_template,
        plan_approval_template,
        progress_tracker_template,
        findings_explorer_template,
        followup_actions_template,
    ):
        assert isinstance(tpl, WidgetTemplate)
        assert tpl.version == "1.0"
        assert tpl.data_schema.get("type") == "object"


def test_widget_templates_safely_escape_special_characters() -> None:
    tricky_query = 'Compare "Postgres" & SQLite\nwith quotes'

    questions = [
        ClarificationQuestion(
            id="q_0",
            question='Pick your "primary" focus:',
            kind="single_select",
            options=[
                ClarificationOption(value="perf", label="Performance"),
                ClarificationOption(value="ops", label="Operations"),
            ],
        ),
        ClarificationQuestion(
            id="q_1",
            question="Select all relevant criteria:",
            kind="multi_select",
            options=[
                ClarificationOption(value="wal", label="WAL concurrency"),
                ClarificationOption(value="backup", label="Backups"),
            ],
        ),
        ClarificationQuestion(
            id="q_2",
            question="Target region?",
            kind="short_text",
            placeholder='e.g. "US-East"',
        ),
        ClarificationQuestion(
            id="q_3",
            question="Share any extra context:",
            kind="long_text",
            placeholder="Describe workload...",
        ),
    ]

    clarify_pending = build_clarifications_widget(
        workflow_id="wf_12345678",
        intro='Help us scope "deep" research.',
        questions=questions,
        state="pending",
    )
    assert clarify_pending.type == "Card"

    clarify_submitted = build_clarifications_widget(
        workflow_id="wf_12345678",
        intro='Help us scope "deep" research.',
        questions=questions,
        state="submitted",
        answers={"q_0": "perf", "q_1": "WAL concurrency, Backups", "q_2": "EU", "q_3": "High write"},
    )
    assert clarify_submitted.type == "Card"

    clarify_skipped = build_clarifications_widget(
        workflow_id="wf_12345678",
        questions=questions,
        state="skipped",
    )
    assert clarify_skipped.type == "Card"

    pending = build_plan_approval_widget(
        workflow_id="wf_12345678",
        sub_questions=[tricky_query, "Backup tradeoffs"],
        state="pending",
    )
    assert pending.type == "Card"

    approved = build_plan_approval_widget(
        workflow_id="wf_12345678",
        sub_questions=[tricky_query],
        state="approved",
        selected_depth="deep",
        extra_focus='Include "MVCC" notes',
    )
    assert approved.type == "Card"

    tracker = build_progress_tracker_widget(
        ResearchProgress(
            workflow_id="wf_12345678",
            query=tricky_query,
            phase=ResearchPhase.RESEARCHING,
            message='Researching "WAL" mode (1/2)...',
            enhanced_prompt='Research "Postgres" vs SQLite focusing on WAL & durability.',
            completed_sub_questions=1,
            total_sub_questions=2,
        )
    )
    assert tracker.type == "Card"
    assert "ENHANCED RESEARCH BRIEF" in tracker.model_dump_json()

    explorer = build_findings_explorer_widget(
        workflow_id="wf_12345678",
        enhanced_prompt='Research "Postgres" vs SQLite focusing on WAL & durability.',
        findings=[
            ResearchFinding(
                sub_question='How does "WAL" mode work?',
                summary='Summary with "quotes"\nand newline.',
                key_points=['Point "A"'],
                sources=["https://example.com/wal"],
            )
        ],
        expanded_indices={0},
    )
    assert explorer.type == "ListView"
    assert "ENHANCED RESEARCH BRIEF" in explorer.model_dump_json()

    followups = build_followup_actions_widget(
        workflow_id="wf_12345678",
        query=tricky_query,
        follow_up_questions=['Deep dive into "WAL" checkpoints', "Compare replication"],
    )
    assert followups.type == "Card"
