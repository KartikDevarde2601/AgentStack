from __future__ import annotations

from pathlib import Path

from chatkit.widgets import WidgetTemplate

from app.domain import ResearchFinding, ResearchPhase, ResearchProgress
from app.widgets import (
    build_findings_explorer_widget,
    build_followup_actions_widget,
    build_plan_approval_widget,
    build_progress_tracker_widget,
    findings_explorer_template,
    followup_actions_template,
    plan_approval_template,
    progress_tracker_template,
)


def test_all_four_widget_templates_are_loaded_from_dot_widget_files() -> None:
    widgets_dir = Path(__file__).resolve().parents[1] / "app" / "widgets"
    expected_files = {
        "research_plan_approval.widget",
        "research_progress_tracker.widget",
        "research_findings_explorer.widget",
        "research_followup_actions.widget",
    }
    actual_files = {p.name for p in widgets_dir.glob("*.widget")}
    assert expected_files == actual_files

    for tpl in (
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
            completed_sub_questions=1,
            total_sub_questions=2,
        )
    )
    assert tracker.type == "Card"

    explorer = build_findings_explorer_widget(
        workflow_id="wf_12345678",
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

    followups = build_followup_actions_widget(
        workflow_id="wf_12345678",
        query=tricky_query,
    )
    assert followups.type == "Card"
