from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from chatkit.server import ChatKitServer
from chatkit.store import Store
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ProgressUpdateEvent,
    Thread,
    ThreadItem,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
)
from temporalio.client import Client, WorkflowHandle

from app.domain import (
    RequestContext,
    ResearchPhase,
    ResearchProgress,
    ResearchRunRecord,
)
from app.temporal.workflows import DeepResearchWorkflow

logger = logging.getLogger(__name__)

ProgressIcon = Literal["clock", "compass", "search", "sparkle", "check-circle", "bug"]

_PHASE_ICONS: dict[ResearchPhase, ProgressIcon] = {
    ResearchPhase.QUEUED: "clock",
    ResearchPhase.PLANNING: "compass",
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


class TemporalResearchRunner:
    def __init__(self, client: Client, task_queue: str) -> None:
        self._client = client
        self._task_queue = task_queue

    async def start(self, workflow_id: str, query: str) -> ResearchWorkflowHandle:
        handle = await self._client.start_workflow(
            DeepResearchWorkflow.run,
            query,
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


_SubscriberItem = ProgressUpdateEvent | AssistantMessageItem | None


@dataclass
class _ActiveWorkflowMonitor:
    thread_id: str
    workflow_id: str
    query: str
    subscribers: set[asyncio.Queue[_SubscriberItem]] = field(default_factory=set)
    task: asyncio.Task[None] | None = None
    final_item: AssistantMessageItem | None = None
    finished: bool = False

    def broadcast(self, event: _SubscriberItem) -> None:
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

    def set_runner(self, runner: ResearchWorkflowRunner | None) -> None:
        self._runner = runner

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
                    monitor.broadcast(
                        ProgressUpdateEvent(
                            icon=_PHASE_ICONS[progress.phase],
                            text=progress.message,
                        )
                    )

                if progress.phase.is_terminal:
                    break

                progress = None
                await asyncio.sleep(self._poll_interval_seconds)

            assistant_item = await self._persist_terminal_item(
                thread_id=monitor.thread_id,
                workflow_id=monitor.workflow_id,
                query=monitor.query,
                progress=progress,
                context=context,
            )
            monitor.final_item = assistant_item
            monitor.finished = True
            monitor.broadcast(assistant_item)
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
                monitor.broadcast(assistant_item)
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

        queue: asyncio.Queue[_SubscriberItem] = asyncio.Queue()
        monitor.subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if isinstance(event, ProgressUpdateEvent):
                    yield event
                elif isinstance(event, AssistantMessageItem):
                    yield ThreadItemDoneEvent(item=event)
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
