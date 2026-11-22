from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from chatkit.server import ChatKitServer
from chatkit.store import NotFoundError, Store
from chatkit.types import (
    Action,
    AssistantMessageContent,
    AssistantMessageItem,
    ProgressUpdateEvent,
    StreamOptions,
    SyncCustomActionResponse,
    Thread,
    ThreadItem,
    ThreadItemDoneEvent,
    ThreadItemReplacedEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageInput,
    UserMessageItem,
    WidgetItem,
)
from temporalio.client import Client, WorkflowHandle

from app.domain import (
    ApprovalDecision,
    RequestContext,
    ResearchFinding,
    ResearchPhase,
    ResearchProgress,
    ResearchRunRecord,
)
from app.temporal.workflows import DeepResearchWorkflow
from app.widgets import (
    APPROVE_PLAN_ACTION,
    FOLLOWUP_ACTION,
    REJECT_PLAN_ACTION,
    TOGGLE_FINDING_ACTION,
    actions_widget_item_id,
    approval_widget_item_id,
    build_findings_explorer_widget,
    build_followup_actions_widget,
    build_plan_approval_widget,
    build_progress_tracker_widget,
    findings_widget_item_id,
    progress_widget_item_id,
)

logger = logging.getLogger(__name__)

ProgressIcon = Literal["clock", "compass", "search", "sparkle", "check-circle", "bug"]

_PHASE_ICONS: dict[ResearchPhase, ProgressIcon] = {
    ResearchPhase.QUEUED: "clock",
    ResearchPhase.PLANNING: "compass",
    ResearchPhase.AWAITING_APPROVAL: "compass",
    ResearchPhase.RESEARCHING: "search",
    ResearchPhase.SYNTHESIZING: "sparkle",
    ResearchPhase.COMPLETED: "check-circle",
    ResearchPhase.FAILED: "bug",
}


class ResearchRunStore(Protocol):
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

    async def list_research_runs(self, thread_id: str) -> list[ResearchRunRecord]: ...

    async def list_active_research_runs(self) -> list[ResearchRunRecord]: ...


class ResearchWorkflowHandle(Protocol):
    workflow_id: str

    async def get_progress(self) -> ResearchProgress: ...


class ResearchWorkflowRunner(Protocol):
    async def start(self, workflow_id: str, query: str) -> ResearchWorkflowHandle: ...

    def get_handle(self, workflow_id: str) -> ResearchWorkflowHandle: ...


class TemporalWorkflowHandleAdapter:
    def __init__(
        self, workflow_id: str, handle: WorkflowHandle[DeepResearchWorkflow, ResearchProgress]
    ) -> None:
        self.workflow_id = workflow_id
        self._handle = handle

    async def get_progress(self) -> ResearchProgress:
        return await self._handle.query(DeepResearchWorkflow.get_progress)

    async def submit_approval(self, decision: ApprovalDecision) -> None:
        await self._handle.signal(DeepResearchWorkflow.submit_approval, decision)


class TemporalResearchRunner:
    def __init__(
        self, client: Client, task_queue: str, require_approval: bool = True
    ) -> None:
        self._client = client
        self._task_queue = task_queue
        self._require_approval = require_approval

    async def start(self, workflow_id: str, query: str) -> ResearchWorkflowHandle:
        handle = await self._client.start_workflow(
            DeepResearchWorkflow.run,
            args=[query, self._require_approval],
            id=workflow_id,
            task_queue=self._task_queue,
        )
        return TemporalWorkflowHandleAdapter(workflow_id=workflow_id, handle=handle)

    def get_handle(self, workflow_id: str) -> ResearchWorkflowHandle:
        handle = self._client.get_workflow_handle(workflow_id)
        return TemporalWorkflowHandleAdapter(workflow_id=workflow_id, handle=handle)


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


def workflow_assistant_item_id(workflow_id: str) -> str:
    return f"msg_{workflow_id}"


def format_terminal_report(query: str, progress: ResearchProgress) -> str:
    if progress.phase == ResearchPhase.COMPLETED and progress.report:
        return progress.report
    return (
        "### Research Failed\n\n"
        f"Unable to complete research for **{query}**.\n\n"
        f"Error: `{progress.error or progress.message}`"
    )


RESUME_RESEARCH_ACTION = "research.resume"


class _PauseForApproval:
    pass


_PAUSE_FOR_APPROVAL = _PauseForApproval()

_SubscriberItem = ThreadStreamEvent | _PauseForApproval | None


@dataclass
class _ActiveWorkflowMonitor:
    thread_id: str
    workflow_id: str
    query: str
    subscribers: set[asyncio.Queue[_SubscriberItem]] = field(default_factory=set)
    task: asyncio.Task[None] | None = None
    last_progress: ProgressUpdateEvent | None = None
    awaiting_approval: bool = False
    progress_widget_created: bool = False
    final_item: AssistantMessageItem | None = None
    finished: bool = False

    def broadcast(self, event: _SubscriberItem) -> None:
        if isinstance(event, ProgressUpdateEvent):
            self.last_progress = event
        for queue in list(self.subscribers):
            queue.put_nowait(event)


class DeepResearchChatKitServer(ChatKitServer[RequestContext]):
    def __init__(
        self,
        store: Store[RequestContext],
        runner: ResearchWorkflowRunner | None,
        run_store: ResearchRunStore | None = None,
        poll_interval_seconds: float = 0.25,
    ) -> None:
        super().__init__(store=store)
        self._runner = runner
        self._run_store = run_store
        self._poll_interval_seconds = poll_interval_seconds
        self._active_monitors: dict[str, _ActiveWorkflowMonitor] = {}
        self._findings_by_workflow: dict[str, list[ResearchFinding]] = {}

    def set_runner(self, runner: ResearchWorkflowRunner | None) -> None:
        self._runner = runner

    async def _build_user_message_item(
        self, input: UserMessageInput, thread: ThreadMetadata, context: RequestContext
    ) -> UserMessageItem:
        item = await super()._build_user_message_item(input, thread, context)
        return item.model_copy(update={"created_at": item.created_at.astimezone(UTC)})

    async def close(self) -> None:
        tasks = [
            monitor.task
            for monitor in list(self._active_monitors.values())
            if monitor.task is not None and not monitor.task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_monitors.clear()

    def get_stream_options(
        self, thread: ThreadMetadata, context: RequestContext
    ) -> StreamOptions:
        # Durable Temporal workflows continue executing in the background, so
        # disable the client Stop button to avoid detaching the live stream.
        return StreamOptions(allow_cancel=False)

    async def handle_stream_cancelled(
        self,
        thread: ThreadMetadata,
        pending_items: list[ThreadItem],
        context: RequestContext,
    ) -> None:
        # Durable Temporal workflows continue executing in the background after
        # a browser tab closes or reloads, so do not inject a cancellation marker
        # that would conflict with the background monitor persisting the report.
        has_active_run = any(
            m.thread_id == thread.id and not m.finished
            for m in self._active_monitors.values()
        )
        if has_active_run:
            return
        await super().handle_stream_cancelled(thread, pending_items, context)

    async def _load_full_thread(self, thread_id: str, context: RequestContext) -> Thread:
        await self.reconcile_thread(thread_id, context=context)
        return await super()._load_full_thread(thread_id, context=context)

    async def reconcile_active_runs(
        self, context: RequestContext | None = None
    ) -> None:
        if self._run_store is None or self._runner is None:
            return
        ctx = context or RequestContext()
        try:
            active_runs = await self._run_store.list_active_research_runs()
        except Exception as exc:
            logger.warning("Failed to list active research runs for reconciliation: %s", exc)
            return
        for run in active_runs:
            await self._reconcile_run(run, context=ctx)

    async def reconcile_thread(
        self, thread_id: str, context: RequestContext | None = None
    ) -> None:
        if self._run_store is None or self._runner is None:
            return
        ctx = context or RequestContext()
        try:
            runs = await self._run_store.list_research_runs(thread_id)
        except Exception as exc:
            logger.warning("Failed to list research runs for thread %s: %s", thread_id, exc)
            return
        for run in runs:
            if not run.phase.is_terminal:
                await self._reconcile_run(run, context=ctx)

    async def _reconcile_run(
        self, run: ResearchRunRecord, *, context: RequestContext
    ) -> None:
        if self._runner is None:
            return
        existing = self._active_monitors.get(run.workflow_id)
        if existing is not None and not existing.finished:
            return

        try:
            handle = self._runner.get_handle(run.workflow_id)
            progress = await handle.get_progress()
        except Exception as exc:
            logger.warning(
                "Unable to query Temporal workflow %s during reconciliation: %s",
                run.workflow_id,
                exc,
            )
            return

        await self._record_run(
            thread_id=run.thread_id,
            workflow_id=run.workflow_id,
            query=run.query,
            phase=progress.phase,
            message=progress.message,
            sub_questions=progress.sub_questions,
            report=progress.report,
            error=progress.error,
        )

        if progress.phase.is_terminal:
            await self._persist_terminal_item(
                thread_id=run.thread_id,
                workflow_id=run.workflow_id,
                query=run.query,
                progress=progress,
                context=context,
            )
            return

        # Still running in Temporal with no local watcher (e.g. after server restart):
        # attach a background monitor so completion persists automatically.
        self._start_background_monitor(
            thread_id=run.thread_id,
            workflow_id=run.workflow_id,
            query=run.query,
            handle=handle,
            context=context,
            initial_progress=progress,
        )

    def _start_background_monitor(
        self,
        *,
        thread_id: str,
        workflow_id: str,
        query: str,
        handle: ResearchWorkflowHandle,
        context: RequestContext,
        initial_progress: ResearchProgress | None = None,
    ) -> _ActiveWorkflowMonitor:
        existing = self._active_monitors.get(workflow_id)
        if existing is not None and not existing.finished:
            return existing

        monitor = _ActiveWorkflowMonitor(
            thread_id=thread_id,
            workflow_id=workflow_id,
            query=query,
        )
        self._active_monitors[workflow_id] = monitor
        monitor.task = asyncio.create_task(
            self._watch_workflow_until_terminal(
                monitor=monitor,
                handle=handle,
                context=context,
                initial_progress=initial_progress,
            ),
            name=f"watch-{workflow_id}",
        )
        return monitor

    async def _watch_workflow_until_terminal(
        self,
        *,
        monitor: _ActiveWorkflowMonitor,
        handle: ResearchWorkflowHandle,
        context: RequestContext,
        initial_progress: ResearchProgress | None = None,
    ) -> None:
        last_signature: tuple[ResearchPhase, str, int] | None = None
        progress: ResearchProgress | None = initial_progress

        try:
            while True:
                if progress is None:
                    progress = await handle.get_progress()

                signature = (
                    progress.phase,
                    progress.message,
                    progress.completed_sub_questions,
                )
                if signature != last_signature:
                    last_signature = signature
                    await self._record_run(
                        thread_id=monitor.thread_id,
                        workflow_id=monitor.workflow_id,
                        query=monitor.query,
                        phase=progress.phase,
                        message=progress.message,
                        sub_questions=progress.sub_questions,
                        report=progress.report,
                        error=progress.error,
                    )
                    if progress.phase == ResearchPhase.AWAITING_APPROVAL:
                        monitor.awaiting_approval = True
                        approval_item = WidgetItem(
                            id=approval_widget_item_id(monitor.workflow_id),
                            thread_id=monitor.thread_id,
                            created_at=datetime.now(UTC),
                            widget=build_plan_approval_widget(
                                workflow_id=monitor.workflow_id,
                                sub_questions=progress.sub_questions,
                                state="pending",
                            ),
                        )
                        await self.store.save_item(
                            monitor.thread_id, approval_item, context
                        )
                        monitor.broadcast(ThreadItemDoneEvent(item=approval_item))
                        monitor.broadcast(_PAUSE_FOR_APPROVAL)
                    else:
                        monitor.awaiting_approval = False
                        monitor.broadcast(
                            ProgressUpdateEvent(
                                icon=_PHASE_ICONS[progress.phase],
                                text=progress.message,
                            )
                        )
                        if progress.phase in (
                            ResearchPhase.RESEARCHING,
                            ResearchPhase.SYNTHESIZING,
                        ):
                            progress_item = WidgetItem(
                                id=progress_widget_item_id(monitor.workflow_id),
                                thread_id=monitor.thread_id,
                                created_at=datetime.now(UTC),
                                widget=build_progress_tracker_widget(progress),
                            )
                            await self.store.save_item(
                                monitor.thread_id, progress_item, context
                            )
                            if not monitor.progress_widget_created:
                                monitor.progress_widget_created = True
                                monitor.broadcast(
                                    ThreadItemDoneEvent(item=progress_item)
                                )
                            else:
                                monitor.broadcast(
                                    ThreadItemReplacedEvent(item=progress_item)
                                )

                if progress.phase.is_terminal:
                    break

                progress = None
                await asyncio.sleep(self._poll_interval_seconds)

            if monitor.progress_widget_created:
                done_progress_item = WidgetItem(
                    id=progress_widget_item_id(monitor.workflow_id),
                    thread_id=monitor.thread_id,
                    created_at=datetime.now(UTC),
                    widget=build_progress_tracker_widget(progress),
                )
                await self.store.save_item(
                    monitor.thread_id, done_progress_item, context
                )
                monitor.broadcast(ThreadItemReplacedEvent(item=done_progress_item))

            if progress.findings:
                self._findings_by_workflow[monitor.workflow_id] = list(
                    progress.findings
                )
                findings_item = WidgetItem(
                    id=findings_widget_item_id(monitor.workflow_id),
                    thread_id=monitor.thread_id,
                    created_at=datetime.now(UTC),
                    widget=build_findings_explorer_widget(
                        workflow_id=monitor.workflow_id,
                        findings=progress.findings,
                    ),
                )
                await self.store.save_item(monitor.thread_id, findings_item, context)
                monitor.broadcast(ThreadItemDoneEvent(item=findings_item))

            assistant_item = await self._persist_terminal_item(
                thread_id=monitor.thread_id,
                workflow_id=monitor.workflow_id,
                query=monitor.query,
                progress=progress,
                context=context,
            )
            monitor.final_item = assistant_item
            monitor.finished = True
            monitor.broadcast(ThreadItemDoneEvent(item=assistant_item))

            was_cancelled = (
                progress.approval_decision is not None
                and not progress.approval_decision.approved
            )
            if progress.phase == ResearchPhase.COMPLETED and not was_cancelled:
                actions_item = WidgetItem(
                    id=actions_widget_item_id(monitor.workflow_id),
                    thread_id=monitor.thread_id,
                    created_at=datetime.now(UTC),
                    widget=build_followup_actions_widget(
                        workflow_id=monitor.workflow_id,
                        query=monitor.query,
                    ),
                )
                await self.store.save_item(monitor.thread_id, actions_item, context)
                monitor.broadcast(ThreadItemDoneEvent(item=actions_item))

            monitor.broadcast(None)
        except asyncio.CancelledError:
            monitor.finished = True
            monitor.broadcast(None)
            raise
        except Exception as exc:
            logger.exception(
                "Background watcher failed for workflow %s: %s",
                monitor.workflow_id,
                exc,
            )
            failed_progress = ResearchProgress(
                workflow_id=monitor.workflow_id,
                query=monitor.query,
                phase=ResearchPhase.FAILED,
                message=f"Deep research failed: {exc}",
                error=str(exc),
            )
            try:
                await self._record_run(
                    thread_id=monitor.thread_id,
                    workflow_id=monitor.workflow_id,
                    query=monitor.query,
                    phase=failed_progress.phase,
                    message=failed_progress.message,
                    error=failed_progress.error,
                )
                assistant_item = await self._persist_terminal_item(
                    thread_id=monitor.thread_id,
                    workflow_id=monitor.workflow_id,
                    query=monitor.query,
                    progress=failed_progress,
                    context=context,
                )
                monitor.final_item = assistant_item
                monitor.broadcast(ThreadItemDoneEvent(item=assistant_item))
            except Exception:
                logger.exception(
                    "Failed to persist failure item for workflow %s",
                    monitor.workflow_id,
                )
            monitor.finished = True
            monitor.broadcast(None)
        finally:
            self._active_monitors.pop(monitor.workflow_id, None)

    async def _persist_terminal_item(
        self,
        *,
        thread_id: str,
        workflow_id: str,
        query: str,
        progress: ResearchProgress,
        context: RequestContext,
    ) -> AssistantMessageItem:
        final_text = format_terminal_report(query, progress)
        item = AssistantMessageItem(
            id=workflow_assistant_item_id(workflow_id),
            thread_id=thread_id,
            created_at=datetime.now(UTC),
            content=[AssistantMessageContent(text=final_text)],
        )
        await self.store.save_item(thread_id, item, context)
        return item

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
                text="Please enter a research question so I can investigate it for you.",
            )
            return

        if not thread.title:
            thread.title = make_thread_title(query)
            await self.store.save_thread(
                ThreadMetadata.model_validate(thread.model_dump(exclude={"items"})),
                context,
            )

        if self._runner is None:
            yield self._assistant_message_event(
                thread=thread,
                context=context,
                text=(
                    "The Temporal workflow service is currently unavailable. "
                    "Please ensure Temporal and the research worker are running and try again."
                ),
            )
            return

        async for event in self._start_and_stream_research(
            thread=thread, query=query, context=context
        ):
            yield event

    async def _start_and_stream_research(
        self,
        *,
        thread: ThreadMetadata,
        query: str,
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        assert self._runner is not None
        workflow_id = f"deep-research-{thread.id}-{uuid.uuid4().hex[:8]}"
        await self._record_run(
            thread_id=thread.id,
            workflow_id=workflow_id,
            query=query,
            phase=ResearchPhase.QUEUED,
            message="Starting deep research workflow...",
        )
        yield ProgressUpdateEvent(
            icon=_PHASE_ICONS[ResearchPhase.QUEUED],
            text="Starting deep research workflow...",
        )

        handle = await self._runner.start(workflow_id=workflow_id, query=query)
        monitor = self._start_background_monitor(
            thread_id=thread.id,
            workflow_id=workflow_id,
            query=query,
            handle=handle,
            context=context,
        )

        async for event in self._stream_monitor(monitor, replay_last_progress=False):
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
            async for event in self._handle_resume_action(thread, payload, context):
                yield event
            return

        if action.type in (APPROVE_PLAN_ACTION, REJECT_PLAN_ACTION):
            async for event in self._handle_approval_action(
                thread=thread,
                approved=(action.type == APPROVE_PLAN_ACTION),
                payload=payload,
                sender=sender,
                context=context,
            ):
                yield event
            return

        if action.type == FOLLOWUP_ACTION:
            async for event in self._handle_followup_action(
                thread=thread,
                payload=payload,
                context=context,
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

        payload = action.payload if isinstance(action.payload, dict) else {}
        workflow_id = str(payload.get("workflow_id") or "")
        if not workflow_id:
            return SyncCustomActionResponse(updated_item=None)

        raw_indices = payload.get("expanded_indices") or []
        expanded_indices = {
            int(i) for i in raw_indices if isinstance(i, int) or str(i).isdigit()
        }

        findings = await self._load_workflow_findings(workflow_id)
        if not findings:
            return SyncCustomActionResponse(updated_item=None)

        updated_item = WidgetItem(
            id=sender.id if sender is not None else findings_widget_item_id(workflow_id),
            thread_id=thread.id,
            created_at=sender.created_at if sender is not None else datetime.now(UTC),
            widget=build_findings_explorer_widget(
                workflow_id=workflow_id,
                findings=findings,
                expanded_indices=expanded_indices,
            ),
        )
        await self.store.save_item(thread.id, updated_item, context)
        return SyncCustomActionResponse(updated_item=updated_item)

    async def _handle_resume_action(
        self,
        thread: ThreadMetadata,
        payload: dict[str, Any],
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        await self.reconcile_thread(thread.id, context=context)

        raw_workflow_id = payload.get("workflow_id")
        requested_workflow_id = (
            raw_workflow_id
            if isinstance(raw_workflow_id, str) and raw_workflow_id
            else None
        )

        monitor = self._find_active_monitor(
            thread_id=thread.id, workflow_id=requested_workflow_id
        )
        if monitor is not None:
            if monitor.awaiting_approval:
                return
            async for event in self._stream_monitor(monitor, replay_last_progress=True):
                yield event
            return

        # Race fallback: the workflow reached a terminal phase before action() ran.
        if self._run_store is None:
            return
        runs = await self._run_store.list_research_runs(thread.id)
        target_run = next(
            (
                run
                for run in runs
                if requested_workflow_id is None
                or run.workflow_id == requested_workflow_id
            ),
            None,
        )
        if target_run is None or not target_run.phase.is_terminal:
            return

        item_id = workflow_assistant_item_id(target_run.workflow_id)
        try:
            persisted_item = await self.store.load_item(thread.id, item_id, context)
        except NotFoundError:
            return
        if isinstance(persisted_item, AssistantMessageItem):
            yield ThreadItemDoneEvent(item=persisted_item)

    async def _handle_approval_action(
        self,
        *,
        thread: ThreadMetadata,
        approved: bool,
        payload: dict[str, Any],
        sender: WidgetItem | None,
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        workflow_id = str(payload.get("workflow_id") or "")
        if not workflow_id or self._runner is None:
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

        raw_depth = str(payload.get("depth") or "standard")
        depth: Literal["concise", "standard", "deep"] = (
            raw_depth if raw_depth in ("concise", "standard", "deep") else "standard"  # type: ignore[assignment]
        )
        extra_focus = str(payload.get("extra_focus") or "").strip()

        decision = ApprovalDecision(
            approved=approved,
            sub_questions=selected_questions,
            depth=depth,
            extra_focus=extra_focus,
            reason=None if approved else "Cancelled from research plan approval card.",
        )

        handle = self._runner.get_handle(workflow_id)
        submit_fn = getattr(handle, "submit_approval", None)
        if callable(submit_fn):
            await submit_fn(decision)

        monitor = self._active_monitors.get(workflow_id)
        if monitor is not None:
            monitor.awaiting_approval = False
        elif self._run_store is not None:
            runs = await self._run_store.list_research_runs(thread.id)
            run = next((r for r in runs if r.workflow_id == workflow_id), None)
            if run is not None:
                monitor = self._start_background_monitor(
                    thread_id=thread.id,
                    workflow_id=workflow_id,
                    query=run.query,
                    handle=handle,
                    context=context,
                )

        effective_questions = list(selected_questions)
        if approved and extra_focus and extra_focus not in effective_questions:
            effective_questions.append(extra_focus)

        resolved_widget = WidgetItem(
            id=sender.id if sender is not None else approval_widget_item_id(workflow_id),
            thread_id=thread.id,
            created_at=sender.created_at if sender is not None else datetime.now(UTC),
            widget=build_plan_approval_widget(
                workflow_id=workflow_id,
                sub_questions=effective_questions,
                state="approved" if approved else "cancelled",
                selected_depth=depth,
                extra_focus=extra_focus,
            ),
        )
        yield ThreadItemReplacedEvent(item=resolved_widget)

        if monitor is not None:
            async for event in self._stream_monitor(monitor, replay_last_progress=False):
                yield event

    async def _handle_followup_action(
        self,
        *,
        thread: ThreadMetadata,
        payload: dict[str, Any],
        context: RequestContext,
    ) -> AsyncIterator[ThreadStreamEvent]:
        workflow_id = str(payload.get("workflow_id") or "")
        mode = str(payload.get("mode") or "")
        query = str(payload.get("query") or "")

        if mode == "deep_dive" and self._runner is not None and query:
            async for event in self._start_and_stream_research(
                thread=thread,
                query=f"Deep-dive follow-up on: {query}",
                context=context,
            ):
                yield event
            return

        findings = await self._load_workflow_findings(workflow_id)
        if mode == "tldr":
            if findings:
                bullets = "\n".join(
                    f"- **{f.sub_question}**: {f.summary}" for f in findings
                )
                text = f"### Executive TL;DR\n\n{bullets}"
            else:
                text = "### Executive TL;DR\n\nNo structured findings were found for this run."
            yield self._assistant_message_event(thread=thread, context=context, text=text)
            return

        if mode == "sources":
            sources: list[str] = []
            for finding in findings:
                for src in finding.sources:
                    if src not in sources:
                        sources.append(src)
            if sources:
                lines = "\n".join(f"{idx + 1}. {src}" for idx, src in enumerate(sources))
                text = f"### Sources Index\n\n{lines}"
            else:
                text = "### Sources Index\n\nNo external sources were recorded for this run."
            yield self._assistant_message_event(thread=thread, context=context, text=text)

    async def _load_workflow_findings(self, workflow_id: str) -> list[ResearchFinding]:
        cached = self._findings_by_workflow.get(workflow_id)
        if cached:
            return cached
        if self._runner is None or not workflow_id:
            return []
        try:
            progress = await self._runner.get_handle(workflow_id).get_progress()
        except Exception:
            return []
        if progress.findings:
            self._findings_by_workflow[workflow_id] = list(progress.findings)
            return list(progress.findings)
        return []

    @staticmethod
    def _is_truthy_checkbox(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "on", "yes"}
        return bool(value)

    def _find_active_monitor(
        self, *, thread_id: str, workflow_id: str | None = None
    ) -> _ActiveWorkflowMonitor | None:
        if workflow_id is not None:
            candidate = self._active_monitors.get(workflow_id)
            if (
                candidate is not None
                and candidate.thread_id == thread_id
                and not candidate.finished
            ):
                return candidate
        for candidate in reversed(list(self._active_monitors.values())):
            if candidate.thread_id == thread_id and not candidate.finished:
                return candidate
        return None

    async def _stream_monitor(
        self,
        monitor: _ActiveWorkflowMonitor,
        *,
        replay_last_progress: bool = False,
    ) -> AsyncIterator[ThreadStreamEvent]:
        queue: asyncio.Queue[_SubscriberItem] = asyncio.Queue()
        monitor.subscribers.add(queue)
        initial_progress = monitor.last_progress if replay_last_progress else None
        is_finished = monitor.finished
        final_item = monitor.final_item
        try:
            if is_finished:
                if final_item is not None:
                    yield ThreadItemDoneEvent(item=final_item)
                return

            if initial_progress is not None:
                yield initial_progress

            while True:
                event = await queue.get()
                if event is None or isinstance(event, _PauseForApproval):
                    break
                yield event
        finally:
            monitor.subscribers.discard(queue)

    async def _record_run(
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
    ) -> None:
        if self._run_store is None:
            return
        await self._run_store.upsert_research_run(
            thread_id=thread_id,
            workflow_id=workflow_id,
            query=query,
            phase=phase,
            message=message,
            sub_questions=sub_questions,
            report=report,
            error=error,
        )

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
