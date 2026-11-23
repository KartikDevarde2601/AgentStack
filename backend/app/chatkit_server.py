from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal

from chatkit.server import ChatKitServer
from chatkit.store import Store
from chatkit.types import (
    Action,
    AssistantMessageContent,
    AssistantMessageContentPartTextDelta,
    AssistantMessageItem,
    ProgressUpdateEvent,
    StreamOptions,
    SyncCustomActionResponse,
    ThreadItem,
    ThreadItemAddedEvent,
    ThreadItemDoneEvent,
    ThreadItemReplacedEvent,
    ThreadItemUpdatedEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageInput,
    UserMessageItem,
    WidgetItem,
)
from pydantic import TypeAdapter

from app.domain import (
    ApprovalDecision,
    IncomingMessage,
    RequestContext,
    ResearchPhase,
)
from app.events import ThreadEventBus, ThreadEventKind
from app.temporal.orchestrator import ThreadOrchestrator
from app.widgets import (
    APPROVE_PLAN_ACTION,
    FOLLOWUP_ACTION,
    REJECT_PLAN_ACTION,
    TOGGLE_FINDING_ACTION,
)

logger = logging.getLogger(__name__)

RESUME_RESEARCH_ACTION = "research.resume"
_THREAD_ITEM_ADAPTER = TypeAdapter(ThreadItem)
_VALID_PROGRESS_ICONS = {"clock", "compass", "search", "sparkle", "check-circle", "bug"}


def extract_user_text(item: UserMessageItem | None) -> str:
    if item is None:
        return ""
    parts: list[str] = []
    for content in item.content:
        text = getattr(content, "text", None)
        if text:
            parts.append(str(text))
    return " ".join(parts).strip()


def make_thread_title(query: str, max_len: int = 60) -> str:
    collapsed = " ".join(query.split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1].rstrip() + "…"


class DeepResearchChatKitServer(ChatKitServer[RequestContext]):
    """Stateless ChatKit adapter: signals AgentThreadWorkflow and tails ThreadEventBus."""

    def __init__(
        self,
        *,
        store: Store[RequestContext],
        orchestrator: ThreadOrchestrator | None,
        event_bus: ThreadEventBus,
    ) -> None:
        super().__init__(store=store)
        self._orchestrator = orchestrator
        self._event_bus = event_bus

    def set_orchestrator(self, orchestrator: ThreadOrchestrator | None) -> None:
        self._orchestrator = orchestrator

    async def close(self) -> None:
        return None

    async def _build_user_message_item(
        self, input: UserMessageInput, thread: ThreadMetadata, context: RequestContext
    ) -> UserMessageItem:
        item = await super()._build_user_message_item(input, thread, context)
        return item.model_copy(update={"created_at": item.created_at.astimezone(UTC)})

    def get_stream_options(
        self, thread: ThreadMetadata, context: RequestContext
    ) -> StreamOptions:
        return StreamOptions(allow_cancel=False)

    async def handle_stream_cancelled(
        self,
        thread: ThreadMetadata,
        pending_items: list[ThreadItem],
        context: RequestContext,
    ) -> None:
        # Durable AgentThreadWorkflow continues executing after a browser tab
        # closes or reloads, so do not inject a cancellation marker here.
        return None

    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        query = extract_user_text(input_user_message)
        if not query:
            yield self._assistant_message_event(
                thread=thread,
                context=context,
                text="Please enter a question or tell me what you'd like to research.",
            )
            return

        if not thread.title:
            thread.title = make_thread_title(query)
            await self.store.save_thread(
                ThreadMetadata.model_validate(thread.model_dump(exclude={"items"})),
                context,
            )

        if self._orchestrator is None:
            yield self._assistant_message_event(
                thread=thread,
                context=context,
                text=(
                    "The Temporal workflow service is currently unavailable. "
                    "Please ensure Temporal and the research worker are running and try again."
                ),
            )
            return

        state = await self._orchestrator.get_state(thread.id)
        if state.phase == ResearchPhase.AWAITING_APPROVAL:
            yield self._assistant_message_event(
                thread=thread,
                context=context,
                text=(
                    "Please approve or cancel the pending research plan above "
                    "before starting a new question."
                ),
            )
            return
        if not state.phase.is_terminal:
            yield self._assistant_message_event(
                thread=thread,
                context=context,
                text=(
                    "A response is already in progress for this thread. "
                    "Please wait for it to finish before sending another message."
                ),
            )
            return

        after_seq = await self._event_bus.latest_seq(thread.id)
        message_id = (
            input_user_message.id
            if input_user_message is not None
            else f"msg_{uuid.uuid4().hex[:12]}"
        )
        yield ProgressUpdateEvent(
            icon="clock",
            text="Reading your request...",
        )
        await self._orchestrator.send_message(
            thread.id,
            IncomingMessage(
                message_id=message_id,
                text=query,
                already_persisted=True,
            ),
        )
        async for event in self._tail_thread_events(thread.id, after_seq=after_seq):
            yield event

    async def action(
        self,
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        payload = action.payload if isinstance(action.payload, dict) else {}

        if action.type == RESUME_RESEARCH_ACTION:
            async for event in self._handle_resume_action(thread):
                yield event
            return

        if action.type in (APPROVE_PLAN_ACTION, REJECT_PLAN_ACTION):
            async for event in self._handle_approval_action(
                thread=thread,
                approved=(action.type == APPROVE_PLAN_ACTION),
                payload=payload,
            ):
                yield event
            return

        if action.type == FOLLOWUP_ACTION:
            async for event in self._handle_followup_action(
                thread=thread,
                payload=payload,
            ):
                yield event
            return

    async def sync_action(
        self,
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: RequestContext,
    ) -> SyncCustomActionResponse:
        if action.type != TOGGLE_FINDING_ACTION:
            return SyncCustomActionResponse(updated_item=None)
        return SyncCustomActionResponse(updated_item=sender)

    async def _handle_resume_action(
        self, thread: ThreadMetadata
    ) -> AsyncIterator[ThreadStreamEvent]:
        if self._orchestrator is None:
            return
        state = await self._orchestrator.get_state(thread.id)
        if state.phase.is_terminal or state.phase == ResearchPhase.AWAITING_APPROVAL:
            return
        async for event in self._tail_thread_events(
            thread.id, after_seq=0, active_turn_id=state.active_turn_id
        ):
            yield event

    async def _handle_approval_action(
        self,
        *,
        thread: ThreadMetadata,
        approved: bool,
        payload: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        if self._orchestrator is None:
            return

        original_questions = [
            str(q) for q in (payload.get("sub_questions") or []) if str(q).strip()
        ]
        checkbox_keys = [k for k in payload if k.startswith("q_")]
        if checkbox_keys and original_questions:
            selected_questions = [
                q
                for idx, q in enumerate(original_questions)
                if self._is_truthy_checkbox(payload.get(f"q_{idx}"))
            ]
            if not selected_questions:
                selected_questions = original_questions
        else:
            selected_questions = original_questions

        raw_depth = str(payload.get("depth") or "standard").lower()
        depth: Literal["concise", "standard", "deep"] = (
            raw_depth if raw_depth in ("concise", "standard", "deep") else "standard"  # type: ignore[assignment]
        )
        extra_focus = str(payload.get("extra_focus") or "").strip()

        decision = ApprovalDecision(
            approved=approved,
            sub_questions=selected_questions,
            depth=depth,
            extra_focus=extra_focus,
            reason=None if approved else "Cancelled by user in approval card.",
        )

        after_seq = await self._event_bus.latest_seq(thread.id)
        await self._orchestrator.submit_approval(thread.id, decision)
        async for event in self._tail_thread_events(thread.id, after_seq=after_seq):
            yield event

    async def _handle_followup_action(
        self,
        *,
        thread: ThreadMetadata,
        payload: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        if self._orchestrator is None:
            return
        mode = str(payload.get("mode") or "executive_summary")
        base_query = str(payload.get("query") or thread.title or "this topic").strip()
        prompts = {
            "executive_summary": f"Provide a concise executive summary and key risks for: {base_query}",
            "deep_dive": f"Do a deeper technical and operational dive into: {base_query}",
            "compare_alternatives": f"Compare the main alternatives and tradeoffs for: {base_query}",
        }
        prompt = prompts.get(mode, f"Follow up on: {base_query}")
        state = await self._orchestrator.get_state(thread.id)
        if not state.phase.is_terminal:
            return
        after_seq = await self._event_bus.latest_seq(thread.id)
        await self._orchestrator.send_message(
            thread.id,
            IncomingMessage(
                message_id=f"msg_{uuid.uuid4().hex[:12]}",
                text=prompt,
                already_persisted=False,
            ),
        )
        async for event in self._tail_thread_events(thread.id, after_seq=after_seq):
            yield event

    async def _tail_thread_events(
        self,
        thread_id: str,
        *,
        after_seq: int,
        active_turn_id: str | None = None,
    ) -> AsyncIterator[ThreadStreamEvent]:
        highest_attempt_by_activity: dict[str, int] = {}

        async for ev in self._event_bus.subscribe(thread_id, after_seq=after_seq):
            if active_turn_id is not None and ev.turn_id != active_turn_id:
                continue

            if ev.activity_id is not None:
                prev_attempt = highest_attempt_by_activity.get(ev.activity_id, 0)
                if ev.attempt < prev_attempt:
                    continue
                is_retry = prev_attempt > 0 and ev.attempt > prev_attempt
                highest_attempt_by_activity[ev.activity_id] = ev.attempt
            else:
                is_retry = False

            if ev.kind == ThreadEventKind.PROGRESS:
                icon = str(ev.payload.get("icon") or "compass")
                if icon not in _VALID_PROGRESS_ICONS:
                    icon = "compass"
                yield ProgressUpdateEvent(
                    icon=icon,  # type: ignore[arg-type]
                    text=str(ev.payload.get("text") or ""),
                )
            elif ev.kind == ThreadEventKind.ITEM_ADDED:
                item = _THREAD_ITEM_ADAPTER.validate_python(ev.payload["item"])
                if is_retry:
                    # Reset previously streamed partial text before retry deltas arrive.
                    yield ThreadItemReplacedEvent(item=item)
                else:
                    yield ThreadItemAddedEvent(item=item)
            elif ev.kind == ThreadEventKind.TEXT_DELTA:
                yield ThreadItemUpdatedEvent(
                    item_id=str(ev.payload["item_id"]),
                    update=AssistantMessageContentPartTextDelta(
                        content_index=0,
                        delta=str(ev.payload.get("delta") or ""),
                    ),
                )
            elif ev.kind == ThreadEventKind.ITEM_DONE:
                item = _THREAD_ITEM_ADAPTER.validate_python(ev.payload["item"])
                yield ThreadItemDoneEvent(item=item)
            elif ev.kind == ThreadEventKind.ITEM_REPLACED:
                item = _THREAD_ITEM_ADAPTER.validate_python(ev.payload["item"])
                yield ThreadItemReplacedEvent(item=item)
            elif ev.kind == ThreadEventKind.APPROVAL_PAUSE:
                return
            elif ev.kind == ThreadEventKind.TURN_END:
                return

    def _assistant_message_event(
        self,
        *,
        thread: ThreadMetadata,
        context: RequestContext,
        text: str,
    ) -> ThreadItemDoneEvent:
        item = AssistantMessageItem(
            id=self.store.generate_item_id("message", thread, context),
            thread_id=thread.id,
            created_at=datetime.now(UTC),
            content=[AssistantMessageContent(text=text)],
        )
        return ThreadItemDoneEvent(item=item)

    @staticmethod
    def _is_truthy_checkbox(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)
