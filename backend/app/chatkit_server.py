from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Literal, Protocol

from chatkit.server import ChatKitServer
from chatkit.store import Store
from chatkit.types import (
    AssistantMessageContent,
    AssistantMessageItem,
    ProgressUpdateEvent,
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


class ResearchWorkflowHandle(Protocol):
    workflow_id: str

    async def get_progress(self) -> ResearchProgress: ...


class ResearchWorkflowRunner(Protocol):
    async def start(self, workflow_id: str, query: str) -> ResearchWorkflowHandle: ...


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

    def set_runner(self, runner: ResearchWorkflowRunner | None) -> None:
        self._runner = runner

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
            await self.store.save_thread(thread, context)

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
        last_signature: tuple[ResearchPhase, str, int] | None = None
        progress: ResearchProgress | None = None

        while True:
            progress = await handle.get_progress()
            signature = (
                progress.phase,
                progress.message,
                progress.completed_sub_questions,
            )
            if signature != last_signature:
                last_signature = signature
                await self._record_run(
                    thread_id=thread.id,
                    workflow_id=workflow_id,
                    query=query,
                    phase=progress.phase,
                    message=progress.message,
                    sub_questions=progress.sub_questions,
                    report=progress.report,
                    error=progress.error,
                )
                yield ProgressUpdateEvent(
                    icon=_PHASE_ICONS[progress.phase],
                    text=progress.message,
                )

            if progress.phase.is_terminal:
                break

            await asyncio.sleep(self._poll_interval_seconds)

        if progress.phase == ResearchPhase.COMPLETED and progress.report:
            final_text = progress.report
        else:
            final_text = (
                "### Research Failed\n\n"
                f"Unable to complete research for **{query}**.\n\n"
                f"Error: `{progress.error or progress.message}`"
            )

        yield self._assistant_message_event(
            thread=thread,
            context=context,
            text=final_text,
        )

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
