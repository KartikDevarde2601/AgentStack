from __future__ import annotations

from datetime import UTC, datetime

from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
)

from app.chatkit_protocol import (
    ThreadEventTranslator,
    build_followup_prompt,
    parse_approval_decision,
    parse_clarification_submission,
)
from app.events import ThreadEvent, ThreadEventKind


def test_parse_clarification_submission_handles_multi_select_and_defaults() -> None:
    payload = {
        "workflow_id": "turn_1",
        "questions": [
            {"id": "q_0", "kind": "single_select"},
            {
                "id": "q_1",
                "kind": "multi_select",
                "options": [
                    {"label": "Latency", "value": "latency"},
                    {"label": "Cost", "value": "cost"},
                    {"label": "Ops", "value": "ops"},
                ],
            },
            {"id": "q_2", "kind": "long_text"},
        ],
        "q_0": "production",
        "q_1__opt__0": "true",
        "q_1__opt__2": True,
        "q_2": "   ",
    }

    sub = parse_clarification_submission(payload, skipped=False)
    assert sub.turn_id == "turn_1"
    assert sub.skipped is False
    assert sub.answers == {
        "q_0": "production",
        "q_1": "Latency, Ops",
        "q_2": "No preference",
    }

    skipped = parse_clarification_submission(payload, skipped=True)
    assert skipped.skipped is True
    assert skipped.answers == {}


def test_parse_approval_decision_filters_checked_questions_and_validates_depth() -> None:
    payload = {
        "workflow_id": "turn_2",
        "sub_questions": ["Q1", "Q2", "Q3"],
        "q_0": True,
        "q_1": False,
        "q_2": "yes",
        "depth": "DEEP",
        "extra_focus": "  Postgres  ",
    }
    decision = parse_approval_decision(payload, approved=True)
    assert decision.turn_id == "turn_2"
    assert decision.approved is True
    assert decision.sub_questions == ["Q1", "Q3"]
    assert decision.depth == "deep"
    assert decision.extra_focus == "Postgres"
    assert decision.reason is None

    rejected = parse_approval_decision(
        {"sub_questions": ["Q1"], "q_0": False, "depth": "invalid"}, approved=False
    )
    assert rejected.approved is False
    assert rejected.sub_questions == ["Q1"]  # falls back when all unchecked
    assert rejected.depth == "standard"
    assert rejected.reason is not None


def test_build_followup_prompt_supports_presets_and_custom() -> None:
    assert "executive summary" in build_followup_prompt(
        {"mode": "executive_summary", "query": "Temporal"}, fallback_query="Fallback"
    ).lower()
    assert (
        build_followup_prompt(
            {"mode": "custom", "prompt": "List top 3 risks"}, fallback_query="Fallback"
        )
        == "List top 3 risks"
    )
    assert (
        build_followup_prompt({"mode": "unknown"}, fallback_query="My Topic")
        == "Follow up on: My Topic"
    )


def test_thread_event_translator_handles_streaming_lifecycle_and_retries() -> None:
    translator = ThreadEventTranslator(active_turn_id="turn_1")
    empty_msg = AssistantMessageItem(
        id="msg_1",
        thread_id="thr_1",
        created_at=datetime.now(UTC),
        content=[],
    ).model_dump(mode="json")
    done_msg = AssistantMessageItem(
        id="msg_1",
        thread_id="thr_1",
        created_at=datetime.now(UTC),
        content=[AssistantMessageContent(text="Final report")],
    ).model_dump(mode="json")

    # Different turn_id is ignored
    events, stop = translator.translate(
        ThreadEvent(
            seq=1,
            thread_id="thr_1",
            turn_id="turn_other",
            kind=ThreadEventKind.TURN_END,
            payload={},
        )
    )
    assert events == []
    assert stop is False

    # Initial streaming assistant item opens content part
    events, stop = translator.translate(
        ThreadEvent(
            seq=2,
            thread_id="thr_1",
            turn_id="turn_1",
            kind=ThreadEventKind.ITEM_ADDED,
            payload={"item": empty_msg},
            activity_id="act_synth",
            attempt=1,
        )
    )
    assert not stop
    assert isinstance(events[0], ThreadItemAddedEvent)
    assert isinstance(events[1], ThreadItemUpdatedEvent)
    assert events[1].update.type == "assistant_message.content_part.added"

    # Retry attempt 2 emits Replaced instead of Added
    events, _ = translator.translate(
        ThreadEvent(
            seq=3,
            thread_id="thr_1",
            turn_id="turn_1",
            kind=ThreadEventKind.ITEM_ADDED,
            payload={"item": empty_msg},
            activity_id="act_synth",
            attempt=2,
        )
    )
    assert isinstance(events[0], ThreadItemReplacedEvent)

    # Stale attempt 1 delta is dropped
    events, _ = translator.translate(
        ThreadEvent(
            seq=4,
            thread_id="thr_1",
            turn_id="turn_1",
            kind=ThreadEventKind.TEXT_DELTA,
            payload={"item_id": "msg_1", "delta": "stale"},
            activity_id="act_synth",
            attempt=1,
        )
    )
    assert events == []

    # Done closes open content part before ThreadItemDoneEvent
    events, stop = translator.translate(
        ThreadEvent(
            seq=5,
            thread_id="thr_1",
            turn_id="turn_1",
            kind=ThreadEventKind.ITEM_DONE,
            payload={"item": done_msg},
            activity_id="act_synth",
            attempt=2,
        )
    )
    assert not stop
    assert isinstance(events[0], ThreadItemUpdatedEvent)
    assert events[0].update.type == "assistant_message.content_part.done"
    assert isinstance(events[1], ThreadItemDoneEvent)

    # TURN_END stops streaming
    events, stop = translator.translate(
        ThreadEvent(
            seq=6,
            thread_id="thr_1",
            turn_id="turn_1",
            kind=ThreadEventKind.TURN_END,
            payload={},
        )
    )
    assert events == []
    assert stop is True
