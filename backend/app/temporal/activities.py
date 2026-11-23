from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    InferenceOptions,
    ThreadItem,
    UserMessageItem,
    UserMessageTextContent,
    WidgetItem,
)
from pydantic import BaseModel, Field
from temporalio import activity

from app.domain import (
    ClarificationQuestion,
    RequestContext,
    ResearchFinding,
    ResearchPhase,
    ResearchProgress,
    ResearchRunRecord,
)
from app.events import ThreadEvent, ThreadEventBus, ThreadEventKind
from app.widgets import (
    actions_widget_item_id,
    approval_widget_item_id,
    build_clarifications_widget,
    build_findings_explorer_widget,
    build_followup_actions_widget,
    build_plan_approval_widget,
    build_progress_tracker_widget,
    clarifications_widget_item_id,
    findings_widget_item_id,
    progress_widget_item_id,
)

_PHASE_ICONS: dict[ResearchPhase, str] = {
    ResearchPhase.IDLE: "clock",
    ResearchPhase.QUEUED: "clock",
    ResearchPhase.AWAITING_CLARIFICATIONS: "compass",
    ResearchPhase.PLANNING: "compass",
    ResearchPhase.AWAITING_APPROVAL: "compass",
    ResearchPhase.RESEARCHING: "search",
    ResearchPhase.SYNTHESIZING: "sparkle",
    ResearchPhase.COMPLETED: "check-circle",
    ResearchPhase.INTERRUPTED: "clock",
    ResearchPhase.FAILED: "bug",
}


class ConversationClient(Protocol):
    async def create_conversation(self) -> str: ...

    async def append_conversation_messages(
        self, conversation_id: str, messages: list[dict[str, str]]
    ) -> None: ...


class ThreadPersistenceStore(Protocol):
    async def get_openai_conversation_id(self, thread_id: str) -> str | None: ...

    async def set_openai_conversation_id(
        self, thread_id: str, conversation_id: str
    ) -> None: ...

    async def save_item(
        self, thread_id: str, item: ThreadItem, context: RequestContext
    ) -> None: ...

    async def upsert_research_run(
        self,
        *,
        thread_id: str,
        workflow_id: str,
        query: str,
        phase: ResearchPhase,
        message: str,
        sub_questions: list[str] | None = None,
        report: str | None = None,
        error: str | None = None,
    ) -> ResearchRunRecord: ...


class RecordFailedTurnsInput(BaseModel):
    conversation_id: str
    turns: list[tuple[str, str]] = Field(default_factory=list)


class PersistUserMessageInput(BaseModel):
    thread_id: str
    turn_id: str
    message_id: str
    text: str


class PublishProgressInput(BaseModel):
    thread_id: str
    turn_id: str
    query: str
    phase: ResearchPhase
    message: str
    enhanced_prompt: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    completed_sub_questions: int = 0
    total_sub_questions: int = 0
    update_progress_widget: bool = False
    is_new_progress_widget: bool = False


class PersistClarificationsWidgetInput(BaseModel):
    thread_id: str
    turn_id: str
    intro: str = ""
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    state: Literal["pending", "submitted", "skipped"] = "pending"
    answers: dict[str, str] = Field(default_factory=dict)


class PersistApprovalWidgetInput(BaseModel):
    thread_id: str
    turn_id: str
    sub_questions: list[str] = Field(default_factory=list)
    state: Literal["pending", "approved", "cancelled"] = "pending"
    selected_depth: Literal["concise", "standard", "deep"] = "standard"
    extra_focus: str = ""


class PublishReportInput(BaseModel):
    thread_id: str
    turn_id: str
    query: str
    enhanced_prompt: str = ""
    findings: list[ResearchFinding] = Field(default_factory=list)
    report: str
    follow_up_questions: list[str] = Field(default_factory=list)
    conversation_id: str | None = None


class PersistFinalItemInput(BaseModel):
    thread_id: str
    turn_id: str
    query: str
    phase: ResearchPhase
    message: str
    text: str
    sub_questions: list[str] = Field(default_factory=list)
    error: str | None = None
    record_research_run: bool = True


class PublishTurnEndInput(BaseModel):
    thread_id: str
    turn_id: str
    phase: ResearchPhase
    message: str = ""


def turn_assistant_item_id(turn_id: str) -> str:
    return f"msg_{turn_id}"


def _activity_meta() -> tuple[str | None, int]:
    try:
        info = activity.info()
        return info.activity_id, info.attempt
    except RuntimeError:
        return None, 1


class ThreadActivities:
    """Durable activities owned by AgentThreadWorkflow."""

    def __init__(
        self,
        *,
        client: ConversationClient,
        store: ThreadPersistenceStore,
        event_bus: ThreadEventBus,
    ) -> None:
        self._client = client
        self._store = store
        self._event_bus = event_bus

    async def _publish(
        self,
        *,
        thread_id: str,
        turn_id: str,
        kind: ThreadEventKind,
        payload: dict[str, Any],
    ) -> int:
        act_id, attempt = _activity_meta()
        return await self._event_bus.publish(
            ThreadEvent(
                thread_id=thread_id,
                turn_id=turn_id,
                kind=kind,
                payload=payload,
                activity_id=act_id,
                attempt=attempt,
            )
        )

    async def _save_and_publish_item(
        self,
        *,
        thread_id: str,
        turn_id: str,
        item: ThreadItem,
        kind: ThreadEventKind = ThreadEventKind.ITEM_DONE,
    ) -> None:
        await self._store.save_item(thread_id, item, RequestContext())
        await self._publish(
            thread_id=thread_id,
            turn_id=turn_id,
            kind=kind,
            payload={"item": item.model_dump(mode="json")},
        )

    @activity.defn(name="ensure_openai_conversation_activity")
    async def ensure_openai_conversation_activity(self, thread_id: str) -> str:
        existing = await self._store.get_openai_conversation_id(thread_id)
        if existing:
            return existing
        conversation_id = await self._client.create_conversation()
        await self._store.set_openai_conversation_id(thread_id, conversation_id)
        return conversation_id

    @activity.defn(name="record_failed_turns_activity")
    async def record_failed_turns_activity(
        self, params: RecordFailedTurnsInput
    ) -> None:
        messages: list[dict[str, str]] = []
        for user_query, error_msg in params.turns:
            messages.append({"role": "user", "content": user_query})
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Previous attempt to research this request failed ({error_msg})."
                    ),
                }
            )
        await self._client.append_conversation_messages(
            params.conversation_id, messages
        )

    @activity.defn(name="persist_user_message_activity")
    async def persist_user_message_activity(
        self, params: PersistUserMessageInput
    ) -> None:
        item = UserMessageItem(
            id=params.message_id,
            thread_id=params.thread_id,
            created_at=datetime.now(UTC),
            content=[UserMessageTextContent(text=params.text)],
            attachments=[],
            inference_options=InferenceOptions(),
        )
        await self._save_and_publish_item(
            thread_id=params.thread_id, turn_id=params.turn_id, item=item
        )

    @activity.defn(name="publish_progress_activity")
    async def publish_progress_activity(self, params: PublishProgressInput) -> None:
        await self._store.upsert_research_run(
            thread_id=params.thread_id,
            workflow_id=params.turn_id,
            query=params.query,
            phase=params.phase,
            message=params.message,
            sub_questions=params.sub_questions,
        )
        await self._publish(
            thread_id=params.thread_id,
            turn_id=params.turn_id,
            kind=ThreadEventKind.PROGRESS,
            payload={
                "icon": _PHASE_ICONS.get(params.phase, "compass"),
                "text": params.message,
                "phase": params.phase.value,
            },
        )
        if params.update_progress_widget:
            progress = ResearchProgress(
                workflow_id=params.turn_id,
                query=params.query,
                phase=params.phase,
                message=params.message,
                enhanced_prompt=params.enhanced_prompt,
                sub_questions=params.sub_questions,
                completed_sub_questions=params.completed_sub_questions,
                total_sub_questions=params.total_sub_questions,
            )
            widget_item = WidgetItem(
                id=progress_widget_item_id(params.turn_id),
                thread_id=params.thread_id,
                created_at=datetime.now(UTC),
                widget=build_progress_tracker_widget(progress),
            )
            await self._save_and_publish_item(
                thread_id=params.thread_id,
                turn_id=params.turn_id,
                item=widget_item,
                kind=(
                    ThreadEventKind.ITEM_DONE
                    if params.is_new_progress_widget
                    else ThreadEventKind.ITEM_REPLACED
                ),
            )

    @activity.defn(name="persist_clarifications_widget_activity")
    async def persist_clarifications_widget_activity(
        self, params: PersistClarificationsWidgetInput
    ) -> str:
        widget_id = clarifications_widget_item_id(params.turn_id)
        clarify_item = WidgetItem(
            id=widget_id,
            thread_id=params.thread_id,
            created_at=datetime.now(UTC),
            widget=build_clarifications_widget(
                workflow_id=params.turn_id,
                intro=params.intro,
                questions=params.questions,
                state=params.state,
                answers=params.answers,
            ),
        )
        await self._save_and_publish_item(
            thread_id=params.thread_id,
            turn_id=params.turn_id,
            item=clarify_item,
            kind=(
                ThreadEventKind.ITEM_DONE
                if params.state == "pending"
                else ThreadEventKind.ITEM_REPLACED
            ),
        )
        if params.state == "pending":
            await self._publish(
                thread_id=params.thread_id,
                turn_id=params.turn_id,
                kind=ThreadEventKind.APPROVAL_PAUSE,
                payload={"widget_item_id": widget_id},
            )
        return widget_id

    @activity.defn(name="persist_approval_widget_activity")
    async def persist_approval_widget_activity(
        self, params: PersistApprovalWidgetInput
    ) -> str:
        widget_id = approval_widget_item_id(params.turn_id)
        approval_item = WidgetItem(
            id=widget_id,
            thread_id=params.thread_id,
            created_at=datetime.now(UTC),
            widget=build_plan_approval_widget(
                workflow_id=params.turn_id,
                sub_questions=params.sub_questions,
                state=params.state,
                selected_depth=params.selected_depth,
                extra_focus=params.extra_focus,
            ),
        )
        await self._save_and_publish_item(
            thread_id=params.thread_id,
            turn_id=params.turn_id,
            item=approval_item,
            kind=(
                ThreadEventKind.ITEM_DONE
                if params.state == "pending"
                else ThreadEventKind.ITEM_REPLACED
            ),
        )
        if params.state == "pending":
            await self._publish(
                thread_id=params.thread_id,
                turn_id=params.turn_id,
                kind=ThreadEventKind.APPROVAL_PAUSE,
                payload={"widget_item_id": widget_id},
            )
        return widget_id

    @activity.defn(name="publish_report_activity")
    async def publish_report_activity(self, params: PublishReportInput) -> str:
        item_id = turn_assistant_item_id(params.turn_id)
        created_at = datetime.now(UTC)

        if params.findings or params.enhanced_prompt:
            findings_item = WidgetItem(
                id=findings_widget_item_id(params.turn_id),
                thread_id=params.thread_id,
                created_at=created_at,
                widget=build_findings_explorer_widget(
                    workflow_id=params.turn_id,
                    findings=params.findings,
                    enhanced_prompt=params.enhanced_prompt,
                ),
            )
            await self._save_and_publish_item(
                thread_id=params.thread_id, turn_id=params.turn_id, item=findings_item
            )

        initial_item = AssistantMessageItem(
            id=item_id,
            thread_id=params.thread_id,
            created_at=created_at,
            content=[],
        )
        await self._publish(
            thread_id=params.thread_id,
            turn_id=params.turn_id,
            kind=ThreadEventKind.ITEM_ADDED,
            payload={"item": initial_item.model_dump(mode="json")},
        )

        report = params.report.strip()
        chunk_size = 240
        for idx in range(0, len(report), chunk_size):
            await self._publish(
                thread_id=params.thread_id,
                turn_id=params.turn_id,
                kind=ThreadEventKind.TEXT_DELTA,
                payload={"item_id": item_id, "delta": report[idx : idx + chunk_size]},
            )

        final_item = AssistantMessageItem(
            id=item_id,
            thread_id=params.thread_id,
            created_at=created_at,
            content=[AssistantMessageContent(text=report)],
        )
        await self._store.upsert_research_run(
            thread_id=params.thread_id,
            workflow_id=params.turn_id,
            query=params.query,
            phase=ResearchPhase.COMPLETED,
            message="Deep research completed.",
            sub_questions=[f.sub_question for f in params.findings],
            report=report,
        )
        if params.conversation_id and report:
            await self._client.append_conversation_messages(
                params.conversation_id, [{"role": "assistant", "content": report}]
            )
        await self._save_and_publish_item(
            thread_id=params.thread_id, turn_id=params.turn_id, item=final_item
        )

        actions_item = WidgetItem(
            id=actions_widget_item_id(params.turn_id),
            thread_id=params.thread_id,
            created_at=datetime.now(UTC),
            widget=build_followup_actions_widget(
                workflow_id=params.turn_id,
                query=params.query,
                follow_up_questions=params.follow_up_questions,
            ),
        )
        await self._save_and_publish_item(
            thread_id=params.thread_id, turn_id=params.turn_id, item=actions_item
        )
        return report

    @activity.defn(name="persist_final_item_activity")
    async def persist_final_item_activity(
        self, params: PersistFinalItemInput
    ) -> None:
        item = AssistantMessageItem(
            id=turn_assistant_item_id(params.turn_id),
            thread_id=params.thread_id,
            created_at=datetime.now(UTC),
            content=[AssistantMessageContent(text=params.text)],
        )
        if params.record_research_run:
            await self._store.upsert_research_run(
                thread_id=params.thread_id,
                workflow_id=params.turn_id,
                query=params.query,
                phase=params.phase,
                message=params.message,
                sub_questions=params.sub_questions,
                report=params.text if params.phase != ResearchPhase.FAILED else None,
                error=params.error,
            )
        await self._save_and_publish_item(
            thread_id=params.thread_id, turn_id=params.turn_id, item=item
        )
        await self._publish(
            thread_id=params.thread_id,
            turn_id=params.turn_id,
            kind=ThreadEventKind.TURN_END,
            payload={"phase": params.phase.value, "message": params.message},
        )

    @activity.defn(name="publish_turn_end_activity")
    async def publish_turn_end_activity(self, params: PublishTurnEndInput) -> None:
        await self._publish(
            thread_id=params.thread_id,
            turn_id=params.turn_id,
            kind=ThreadEventKind.TURN_END,
            payload={"phase": params.phase.value, "message": params.message},
        )

    @activity.defn(name="prune_thread_events_activity")
    async def prune_thread_events_activity(self, thread_id: str) -> int:
        thread_removed = await self._event_bus.prune(thread_id=thread_id)
        stale_removed = await self._event_bus.prune(older_than_hours=24)
        return thread_removed + stale_removed
