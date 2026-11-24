from __future__ import annotations

from typing import Any, Literal

from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageContentPartAdded,
    AssistantMessageContentPartDone,
    AssistantMessageContentPartTextDelta,
    AssistantMessageItem,
    ProgressUpdateEvent,
    ThreadItem,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
    ThreadStreamEvent,
)
from pydantic import TypeAdapter

from app.domain import ApprovalDecision, ClarificationSubmission
from app.events import ThreadEvent, ThreadEventKind

_THREAD_ITEM_ADAPTER = TypeAdapter(ThreadItem)
_VALID_PROGRESS_ICONS = {"clock", "compass", "search", "sparkle", "check-circle", "bug"}
_FOLLOWUP_PROMPT_TEMPLATES: dict[str, str] = {
    "executive_summary": "Provide a concise executive summary and key risks for: {query}",
    "deep_dive": "Do a deeper technical and operational dive into: {query}",
    "compare_alternatives": "Compare the main alternatives and tradeoffs for: {query}",
}


def is_truthy_checkbox(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def parse_clarification_submission(
    payload: dict[str, Any], *, skipped: bool
) -> ClarificationSubmission:
    answers: dict[str, str] = {}
    if not skipped:
        for idx, raw_q in enumerate(payload.get("questions") or []):
            if not isinstance(raw_q, dict):
                continue
            q_id = str(raw_q.get("id") or f"q_{idx}")
            q_kind = str(raw_q.get("kind") or "short_text")
            if q_kind == "multi_select":
                answers[q_id] = _parse_multi_select_answer(payload, q_id, raw_q)
            else:
                val = str(payload.get(q_id) or "").strip()
                answers[q_id] = val or "No preference"

    turn_id = str(payload.get("workflow_id") or "").strip() or None
    return ClarificationSubmission(turn_id=turn_id, skipped=skipped, answers=answers)


def _parse_multi_select_answer(
    payload: dict[str, Any], q_id: str, raw_question: dict[str, Any]
) -> str:
    selected: list[str] = []
    for opt_idx, opt in enumerate(raw_question.get("options") or []):
        if not is_truthy_checkbox(payload.get(f"{q_id}__opt__{opt_idx}")):
            continue
        label = (
            str(opt.get("label") or opt.get("value") or "")
            if isinstance(opt, dict)
            else str(opt)
        )
        if label:
            selected.append(label)
    return ", ".join(selected) if selected else "No preference"


def parse_approval_decision(
    payload: dict[str, Any], *, approved: bool
) -> ApprovalDecision:
    original_questions = [
        str(q) for q in (payload.get("sub_questions") or []) if str(q).strip()
    ]
    has_checkboxes = any(k.startswith("q_") for k in payload)
    if has_checkboxes and original_questions:
        selected_questions = [
            q
            for idx, q in enumerate(original_questions)
            if is_truthy_checkbox(payload.get(f"q_{idx}"))
        ] or original_questions
    else:
        selected_questions = original_questions

    raw_depth = str(payload.get("depth") or "standard").lower()
    depth: Literal["concise", "standard", "deep"] = (
        raw_depth if raw_depth in ("concise", "standard", "deep") else "standard"  # type: ignore[assignment]
    )
    extra_focus = str(payload.get("extra_focus") or "").strip()
    turn_id = str(payload.get("workflow_id") or "").strip() or None

    return ApprovalDecision(
        turn_id=turn_id,
        approved=approved,
        sub_questions=selected_questions,
        depth=depth,
        extra_focus=extra_focus,
        reason=None if approved else "Cancelled by user in approval card.",
    )


def build_followup_prompt(payload: dict[str, Any], *, fallback_query: str) -> str:
    mode = str(payload.get("mode") or "executive_summary")
    base_query = str(payload.get("query") or fallback_query or "this topic").strip()
    custom_prompt = str(payload.get("prompt") or "").strip()
    if mode == "custom" and custom_prompt:
        return custom_prompt
    template = _FOLLOWUP_PROMPT_TEMPLATES.get(mode, "Follow up on: {query}")
    return template.format(query=base_query)


class ThreadEventTranslator:
    """Translates internal ThreadEvents into ChatKit ThreadStreamEvents.

    Tracks per-activity retry attempts (dropping stale attempts and replacing
    partial items on retry) and open assistant message content parts so ChatKit
    streams deltas without re-rendering.
    """

    def __init__(self, *, active_turn_id: str | None = None) -> None:
        self._active_turn_id = active_turn_id
        self._highest_attempt_by_activity: dict[str, int] = {}
        self._open_content_part_items: set[str] = set()

    def translate(self, ev: ThreadEvent) -> tuple[list[ThreadStreamEvent], bool]:
        """Return (chatkit_events, should_stop) for a single ThreadEvent."""
        if self._active_turn_id is not None and ev.turn_id != self._active_turn_id:
            return [], False

        is_retry, skip = self._track_activity_attempt(ev)
        if skip:
            return [], False

        match ev.kind:
            case ThreadEventKind.PROGRESS:
                return [self._translate_progress(ev)], False
            case ThreadEventKind.ITEM_ADDED:
                return self._translate_item_added(ev, is_retry=is_retry), False
            case ThreadEventKind.TEXT_DELTA:
                return [self._translate_text_delta(ev)], False
            case ThreadEventKind.ITEM_DONE:
                return self._translate_item_done(ev), False
            case ThreadEventKind.ITEM_REPLACED:
                item = _THREAD_ITEM_ADAPTER.validate_python(ev.payload["item"])
                return [ThreadItemReplacedEvent(item=item)], False
            case ThreadEventKind.APPROVAL_PAUSE | ThreadEventKind.TURN_END:
                return [], True
            case _:
                return [], False

    def _track_activity_attempt(self, ev: ThreadEvent) -> tuple[bool, bool]:
        if ev.activity_id is None:
            return False, False
        prev_attempt = self._highest_attempt_by_activity.get(ev.activity_id, 0)
        if ev.attempt < prev_attempt:
            return False, True
        is_retry = prev_attempt > 0 and ev.attempt > prev_attempt
        self._highest_attempt_by_activity[ev.activity_id] = ev.attempt
        return is_retry, False

    def _translate_progress(self, ev: ThreadEvent) -> ProgressUpdateEvent:
        icon = str(ev.payload.get("icon") or "compass")
        if icon not in _VALID_PROGRESS_ICONS:
            icon = "compass"
        return ProgressUpdateEvent(
            icon=icon,  # type: ignore[arg-type]
            text=str(ev.payload.get("text") or ""),
        )

    def _translate_item_added(
        self, ev: ThreadEvent, *, is_retry: bool
    ) -> list[ThreadStreamEvent]:
        item = _THREAD_ITEM_ADAPTER.validate_python(ev.payload["item"])
        is_streaming_assistant = isinstance(item, AssistantMessageItem) and (
            not item.content or (len(item.content) == 1 and not item.content[0].text)
        )
        if is_streaming_assistant:
            item = item.model_copy(update={"content": []})

        out: list[ThreadStreamEvent] = [
            ThreadItemReplacedEvent(item=item)
            if is_retry
            else ThreadItemAddedEvent(item=item)
        ]
        if is_streaming_assistant:
            self._open_content_part_items.add(item.id)
            out.append(
                ThreadItemUpdatedEvent(
                    item_id=item.id,
                    update=AssistantMessageContentPartAdded(
                        content_index=0,
                        content=AssistantMessageContent(text=""),
                    ),
                )
            )
        return out

    def _translate_text_delta(self, ev: ThreadEvent) -> ThreadItemUpdatedEvent:
        return ThreadItemUpdatedEvent(
            item_id=str(ev.payload["item_id"]),
            update=AssistantMessageContentPartTextDelta(
                content_index=0,
                delta=str(ev.payload.get("delta") or ""),
            ),
        )

    def _translate_item_done(self, ev: ThreadEvent) -> list[ThreadStreamEvent]:
        item = _THREAD_ITEM_ADAPTER.validate_python(ev.payload["item"])
        out: list[ThreadStreamEvent] = []
        if (
            isinstance(item, AssistantMessageItem)
            and item.id in self._open_content_part_items
        ):
            self._open_content_part_items.discard(item.id)
            if item.content:
                out.append(
                    ThreadItemUpdatedEvent(
                        item_id=item.id,
                        update=AssistantMessageContentPartDone(
                            content_index=0,
                            content=item.content[0],
                        ),
                    )
                )
        out.append(ThreadItemDoneEvent(item=item))
        return out
