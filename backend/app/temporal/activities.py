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
    RequestContext,
    ResearchFinding,
    ResearchPhase,
    ResearchPlan,
    ResearchProgress,
    ResearchRunRecord,
)
from app.events import ThreadEvent, ThreadEventBus, ThreadEventKind
from app.research.openai_client import ResearchLLMClient
from app.widgets import (
    actions_widget_item_id,
    approval_widget_item_id,
    build_findings_explorer_widget,
    build_followup_actions_widget,
    build_plan_approval_widget,
    build_progress_tracker_widget,
    findings_widget_item_id,
    progress_widget_item_id,
)

_PHASE_ICONS: dict[ResearchPhase, str] = {
    ResearchPhase.IDLE: "clock",
    ResearchPhase.QUEUED: "clock",
    ResearchPhase.PLANNING: "compass",
    ResearchPhase.AWAITING_APPROVAL: "compass",
    ResearchPhase.RESEARCHING: "search",
    ResearchPhase.SYNTHESIZING: "sparkle",
    ResearchPhase.COMPLETED: "check-circle",
    ResearchPhase.INTERRUPTED: "clock",
    ResearchPhase.FAILED: "bug",
}


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


class CloseToolCallsInput(BaseModel):
    conversation_id: str
    call_ids: list[str] = Field(default_factory=list)


class RecordFailedTurnsInput(BaseModel):
    conversation_id: str
    turns: list[tuple[str, str]] = Field(default_factory=list)


class PersistUserMessageInput(BaseModel):
    thread_id: str
    turn_id: str
    message_id: str
    text: str


class PlanResearchInput(BaseModel):
    query: str
    conversation_id: str | None = None


class GatherFindingInput(BaseModel):
    query: str
    sub_question: str
    depth: str = "standard"
    conversation_id: str | None = None


class SynthesizeReportInput(BaseModel):
    thread_id: str
    turn_id: str
    query: str
    findings: list[ResearchFinding]
    conversation_id: str | None = None


class PublishProgressInput(BaseModel):
    thread_id: str
    turn_id: str
    query: str
    phase: ResearchPhase
    message: str
    sub_questions: list[str] = Field(default_factory=list)
    completed_sub_questions: int = 0
    total_sub_questions: int = 0
    update_progress_widget: bool = False
    is_new_progress_widget: bool = False


class PersistApprovalWidgetInput(BaseModel):
    thread_id: str
    turn_id: str
    sub_questions: list[str] = Field(default_factory=list)
    state: Literal["pending", "approved", "cancelled"] = "pending"
    selected_depth: Literal["concise", "standard", "deep"] = "standard"
    extra_focus: str = ""


class PersistFinalItemInput(BaseModel):
    thread_id: str
    turn_id: str
    query: str
    phase: ResearchPhase
    message: str
    text: str
    sub_questions: list[str] = Field(default_factory=list)
    error: str | None = None


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
    except Exception:
        return None, 1


class ThreadActivities:
    """Durable activities owned by AgentThreadWorkflow."""

    def __init__(
        self,
        *,
        client: ResearchLLMClient,
        store: ThreadPersistenceStore,
        event_bus: ThreadEventBus,
    ) -> None:
        self._client = client
        self._store = store
        self._event_bus = event_bus

    @activity.defn(name="ensure_openai_conversation_activity")
    async def ensure_openai_conversation_activity(self, thread_id: str) -> str:
        existing = await self._store.get_openai_conversation_id(thread_id)
        if existing:
            return existing
        conversation_id = await self._client.create_conversation()
        await self._store.set_openai_conversation_id(thread_id, conversation_id)
        return conversation_id

    @activity.defn(name="close_dangling_tool_calls_activity")
    async def close_dangling_tool_calls_activity(
        self, params: CloseToolCallsInput
    ) -> None:
        await self._client.close_dangling_tool_calls(
            params.conversation_id, params.call_ids
        )

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
        await self._store.save_item(params.thread_id, item, RequestContext())
        act_id, attempt = _activity_meta()
        await self._event_bus.publish(
            ThreadEvent(
                thread_id=params.thread_id,
                turn_id=params.turn_id,
                kind=ThreadEventKind.ITEM_DONE,
                payload={"item": item.model_dump(mode="json")},
                activity_id=act_id,
                attempt=attempt,
            )
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
        act_id, attempt = _activity_meta()
        await self._event_bus.publish(
            ThreadEvent(
                thread_id=params.thread_id,
                turn_id=params.turn_id,
                kind=ThreadEventKind.PROGRESS,
                payload={
                    "icon": _PHASE_ICONS.get(params.phase, "compass"),
                    "text": params.message,
                    "phase": params.phase.value,
                },
                activity_id=act_id,
                attempt=attempt,
            )
        )
        if params.update_progress_widget:
            progress = ResearchProgress(
                workflow_id=params.turn_id,
                query=params.query,
                phase=params.phase,
                message=params.message,
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
            await self._store.save_item(params.thread_id, widget_item, RequestContext())
            await self._event_bus.publish(
                ThreadEvent(
                    thread_id=params.thread_id,
                    turn_id=params.turn_id,
                    kind=(
                        ThreadEventKind.ITEM_DONE
                        if params.is_new_progress_widget
                        else ThreadEventKind.ITEM_REPLACED
                    ),
                    payload={"item": widget_item.model_dump(mode="json")},
                    activity_id=act_id,
                    attempt=attempt,
                )
            )

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
        await self._store.save_item(params.thread_id, approval_item, RequestContext())
        act_id, attempt = _activity_meta()
        kind = (
            ThreadEventKind.ITEM_DONE
            if params.state == "pending"
            else ThreadEventKind.ITEM_REPLACED
        )
        await self._event_bus.publish(
            ThreadEvent(
                thread_id=params.thread_id,
                turn_id=params.turn_id,
                kind=kind,
                payload={"item": approval_item.model_dump(mode="json")},
                activity_id=act_id,
                attempt=attempt,
            )
        )
        if params.state == "pending":
            await self._event_bus.publish(
                ThreadEvent(
                    thread_id=params.thread_id,
                    turn_id=params.turn_id,
                    kind=ThreadEventKind.APPROVAL_PAUSE,
                    payload={"widget_item_id": widget_id},
                    activity_id=act_id,
                    attempt=attempt,
                )
            )
        return widget_id

    @activity.defn(name="plan_research_activity")
    async def plan_research_activity(self, params: PlanResearchInput) -> ResearchPlan:
        return await self._client.generate_research_plan(
            params.query, conversation_id=params.conversation_id
        )

    @activity.defn(name="investigate_sub_question_activity")
    async def investigate_sub_question_activity(
        self, params: GatherFindingInput
    ) -> ResearchFinding:
        return await self._client.investigate_sub_question(
            params.query,
            params.sub_question,
            depth=params.depth,
            conversation_id=params.conversation_id,
        )

    @activity.defn(name="synthesize_report_activity")
    async def synthesize_report_activity(self, params: SynthesizeReportInput) -> str:
        act_id, attempt = _activity_meta()
        item_id = turn_assistant_item_id(params.turn_id)
        created_at = datetime.now(UTC)

        # Publish findings explorer widget before streaming the final report.
        if params.findings:
            findings_item = WidgetItem(
                id=findings_widget_item_id(params.turn_id),
                thread_id=params.thread_id,
                created_at=created_at,
                widget=build_findings_explorer_widget(
                    workflow_id=params.turn_id,
                    findings=params.findings,
                ),
            )
            await self._store.save_item(
                params.thread_id, findings_item, RequestContext()
            )
            await self._event_bus.publish(
                ThreadEvent(
                    thread_id=params.thread_id,
                    turn_id=params.turn_id,
                    kind=ThreadEventKind.ITEM_DONE,
                    payload={"item": findings_item.model_dump(mode="json")},
                    activity_id=act_id,
                    attempt=attempt,
                )
            )

        initial_item = AssistantMessageItem(
            id=item_id,
            thread_id=params.thread_id,
            created_at=created_at,
            content=[AssistantMessageContent(text="")],
        )
        await self._event_bus.publish(
            ThreadEvent(
                thread_id=params.thread_id,
                turn_id=params.turn_id,
                kind=ThreadEventKind.ITEM_ADDED,
                payload={"item": initial_item.model_dump(mode="json")},
                activity_id=act_id,
                attempt=attempt,
            )
        )

        async def _emit_delta(delta: str) -> None:
            await self._event_bus.publish(
                ThreadEvent(
                    thread_id=params.thread_id,
                    turn_id=params.turn_id,
                    kind=ThreadEventKind.TEXT_DELTA,
                    payload={"item_id": item_id, "delta": delta},
                    activity_id=act_id,
                    attempt=attempt,
                )
            )

        report = await self._client.synthesize_report_stream(
            params.query,
            params.findings,
            conversation_id=params.conversation_id,
            on_delta=_emit_delta,
        )

        final_item = AssistantMessageItem(
            id=item_id,
            thread_id=params.thread_id,
            created_at=created_at,
            content=[AssistantMessageContent(text=report)],
        )
        await self._store.save_item(params.thread_id, final_item, RequestContext())
        await self._store.upsert_research_run(
            thread_id=params.thread_id,
            workflow_id=params.turn_id,
            query=params.query,
            phase=ResearchPhase.COMPLETED,
            message="Deep research completed.",
            sub_questions=[f.sub_question for f in params.findings],
            report=report,
        )
        await self._event_bus.publish(
            ThreadEvent(
                thread_id=params.thread_id,
                turn_id=params.turn_id,
                kind=ThreadEventKind.ITEM_DONE,
                payload={"item": final_item.model_dump(mode="json")},
                activity_id=act_id,
                attempt=attempt,
            )
        )

        actions_item = WidgetItem(
            id=actions_widget_item_id(params.turn_id),
            thread_id=params.thread_id,
            created_at=datetime.now(UTC),
            widget=build_followup_actions_widget(
                workflow_id=params.turn_id,
                query=params.query,
            ),
        )
        await self._store.save_item(params.thread_id, actions_item, RequestContext())
        await self._event_bus.publish(
            ThreadEvent(
                thread_id=params.thread_id,
                turn_id=params.turn_id,
                kind=ThreadEventKind.ITEM_DONE,
                payload={"item": actions_item.model_dump(mode="json")},
                activity_id=act_id,
                attempt=attempt,
            )
        )
        return report

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

    @activity.defn(name="persist_interrupted_or_final_activity")
    async def persist_interrupted_or_final_activity(
        self, params: PersistFinalItemInput
    ) -> None:
        final_text = params.text
        if params.phase == ResearchPhase.INTERRUPTED:
            streamed = (
                await self._event_bus.get_turn_streamed_text(
                    params.thread_id, params.turn_id
                )
            ).strip()
            if streamed:
                final_text = (
                    f"{streamed}\n\n---\n"
                    "_Interrupted by a new steering instruction._"
                )

        item = AssistantMessageItem(
            id=turn_assistant_item_id(params.turn_id),
            thread_id=params.thread_id,
            created_at=datetime.now(UTC),
            content=[AssistantMessageContent(text=final_text)],
        )
        await self._store.save_item(params.thread_id, item, RequestContext())
        await self._store.upsert_research_run(
            thread_id=params.thread_id,
            workflow_id=params.turn_id,
            query=params.query,
            phase=params.phase,
            message=params.message,
            sub_questions=params.sub_questions,
            report=final_text if params.phase != ResearchPhase.FAILED else None,
            error=params.error,
        )
        await self._publish(
            thread_id=params.thread_id,
            turn_id=params.turn_id,
            kind=ThreadEventKind.ITEM_DONE,
            payload={"item": item.model_dump(mode="json")},
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
