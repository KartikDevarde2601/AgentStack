from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from chatkit.actions import Action
from chatkit.server import ChatKitServer
from chatkit.store import Store
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ProgressUpdateEvent,
    StreamOptions,
    SyncCustomActionResponse,
    ThreadItem,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageInput,
    UserMessageItem,
    WidgetItem,
)

from app.chatkit_protocol import (
    ThreadEventTranslator,
    build_followup_prompt,
    parse_approval_decision,
    parse_clarification_submission,
)
from app.domain import IncomingMessage, RequestContext, ResearchPhase
from app.events import ThreadEventBus
from app.temporal.client import TemporalUnavailableError
from app.temporal.orchestrator import ThreadOrchestrator, WorkflowActionRejectedError
from app.widgets import (
    APPROVE_PLAN_ACTION,
    FOLLOWUP_ACTION,
    REJECT_PLAN_ACTION,
    RESUME_RESEARCH_ACTION,
    SKIP_CLARIFICATIONS_ACTION,
    SUBMIT_CLARIFICATIONS_ACTION,
    TOGGLE_FINDING_ACTION,
)

_TEMPORAL_UNAVAILABLE_MESSAGE = (
    "The Temporal workflow service is currently unavailable. "
    "Please ensure Temporal and the research worker are running and try again."
)


def extract_user_text(item: UserMessageItem | None) -> str:
    if item is None:
        return ""
    parts = [
        str(text)
        for content in item.content
        if (text := getattr(content, "text", None))
    ]
    return " ".join(parts).strip()


def make_thread_title(query: str, max_len: int = 60) -> str:
    collapsed = " ".join(query.split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1].rstrip() + "…"


def _busy_message_for_phase(phase: ResearchPhase) -> str | None:
    if phase == ResearchPhase.AWAITING_CLARIFICATIONS:
        return (
            "Please submit or skip the clarifying questions card above "
            "before starting a new message."
        )
    if phase == ResearchPhase.AWAITING_APPROVAL:
        return (
            "Please approve or cancel the pending research plan above "
            "before starting a new question."
        )
    if not phase.is_terminal:
        return (
            "A response is already in progress for this thread. "
            "Please wait for it to finish before sending another message."
        )
    return None


class DeepResearchChatKitServer(ChatKitServer[RequestContext]):
    """Stateless ChatKit adapter: signals AgentThreadWorkflow and tails ThreadEventBus."""

    def __init__(
        self,
        *,
        store: Store[RequestContext],
        orchestrator: ThreadOrchestrator,
        event_bus: ThreadEventBus,
    ) -> None:
        super().__init__(store=store)
        self._orchestrator = orchestrator
        self._event_bus = event_bus

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

        try:
            state = await self._orchestrator.get_state(thread.id)
        except TemporalUnavailableError:
            yield self._assistant_message_event(
                thread=thread, context=context, text=_TEMPORAL_UNAVAILABLE_MESSAGE
            )
            return

        if busy_message := _busy_message_for_phase(state.phase):
            yield self._assistant_message_event(
                thread=thread, context=context, text=busy_message
            )
            return

        message_id = (
            input_user_message.id
            if input_user_message is not None
            else f"msg_{uuid.uuid4().hex[:12]}"
        )
        yield ProgressUpdateEvent(icon="clock", text="Reading your request...")
        async for event in self._dispatch_and_stream(
            thread=thread,
            context=context,
            dispatch=lambda: self._orchestrator.send_message(
                thread.id,
                IncomingMessage(
                    message_id=message_id, text=query, already_persisted=True
                ),
            ),
        ):
            yield event

    async def action(
        self,
        thread: ThreadMetadata,
        action: Action[str, Any],
        sender: WidgetItem | None,
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        payload = action.payload if isinstance(action.payload, dict) else {}

        match action.type:
            case _ if action.type == RESUME_RESEARCH_ACTION:
                stream = self._handle_resume_action(thread)
            case _ if action.type in (
                SUBMIT_CLARIFICATIONS_ACTION,
                SKIP_CLARIFICATIONS_ACTION,
            ):
                submission = parse_clarification_submission(
                    payload, skipped=(action.type == SKIP_CLARIFICATIONS_ACTION)
                )
                stream = self._dispatch_and_stream(
                    thread=thread,
                    context=context,
                    dispatch=lambda: self._orchestrator.submit_clarifications(
                        thread.id, submission
                    ),
                )
            case _ if action.type in (APPROVE_PLAN_ACTION, REJECT_PLAN_ACTION):
                decision = parse_approval_decision(
                    payload, approved=(action.type == APPROVE_PLAN_ACTION)
                )
                stream = self._dispatch_and_stream(
                    thread=thread,
                    context=context,
                    dispatch=lambda: self._orchestrator.submit_approval(
                        thread.id, decision
                    ),
                )
            case _ if action.type == FOLLOWUP_ACTION:
                stream = self._handle_followup_action(
                    thread=thread, payload=payload, context=context
                )
            case _:
                return

        async for event in stream:
            yield event

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
        try:
            state = await self._orchestrator.get_state(thread.id)
        except TemporalUnavailableError:
            return
        if state.phase.is_terminal or state.phase in (
            ResearchPhase.AWAITING_CLARIFICATIONS,
            ResearchPhase.AWAITING_APPROVAL,
        ):
            return
        async for event in self._tail_thread_events(
            thread.id, after_seq=0, active_turn_id=state.active_turn_id
        ):
            yield event

    async def _handle_followup_action(
        self,
        *,
        thread: ThreadMetadata,
        payload: dict[str, Any],
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        prompt = build_followup_prompt(
            payload, fallback_query=thread.title or "this topic"
        )
        try:
            state = await self._orchestrator.get_state(thread.id)
        except TemporalUnavailableError:
            yield self._assistant_message_event(
                thread=thread, context=context, text=_TEMPORAL_UNAVAILABLE_MESSAGE
            )
            return
        if not state.phase.is_terminal:
            return
        async for event in self._dispatch_and_stream(
            thread=thread,
            context=context,
            dispatch=lambda: self._orchestrator.send_message(
                thread.id,
                IncomingMessage(
                    message_id=f"msg_{uuid.uuid4().hex[:12]}",
                    text=prompt,
                    already_persisted=False,
                ),
            ),
        ):
            yield event

    async def _dispatch_and_stream(
        self,
        *,
        thread: ThreadMetadata,
        context: RequestContext,
        dispatch: Callable[[], Awaitable[Any]],
    ) -> AsyncIterator[ThreadStreamEvent]:
        after_seq = await self._event_bus.latest_seq(thread.id)
        try:
            await dispatch()
        except TemporalUnavailableError:
            yield self._assistant_message_event(
                thread=thread, context=context, text=_TEMPORAL_UNAVAILABLE_MESSAGE
            )
            return
        except WorkflowActionRejectedError as exc:
            yield self._assistant_message_event(
                thread=thread, context=context, text=str(exc)
            )
            return
        async for event in self._tail_thread_events(thread.id, after_seq=after_seq):
            yield event

    async def _tail_thread_events(
        self,
        thread_id: str,
        *,
        after_seq: int,
        active_turn_id: str | None = None,
    ) -> AsyncIterator[ThreadStreamEvent]:
        translator = ThreadEventTranslator(active_turn_id=active_turn_id)
        async for ev in self._event_bus.subscribe(thread_id, after_seq=after_seq):
            events, should_stop = translator.translate(ev)
            for out_event in events:
                yield out_event
            if should_stop:
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
