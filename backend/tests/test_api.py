from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime

import httpx
import pytest
from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata

from app.config import Settings
from app.domain import (
    ApprovalDecision,
    RequestContext,
    ResearchFinding,
    ResearchPhase,
    ResearchProgress,
    ResearchRunRecord,
    ThreadSummary,
)
from app.main import create_app


class InMemoryChatKitStore(Store[RequestContext]):
    """In-memory implementation of the PostgresChatKitStore contract for API tests."""

    def __init__(self) -> None:
        self._threads: dict[str, tuple[ThreadMetadata, datetime]] = {}
        self._items: dict[str, list[ThreadItem]] = {}
        self._attachments: dict[str, Attachment] = {}
        self._runs: dict[str, ResearchRunRecord] = {}

    async def ping(self) -> bool:
        return True

    async def load_thread(
        self, thread_id: str, context: RequestContext
    ) -> ThreadMetadata:
        entry = self._threads.get(thread_id)
        if entry is None:
            raise NotFoundError(f"Thread {thread_id} not found")
        return entry[0].model_copy(deep=True)

    async def save_thread(
        self, thread: ThreadMetadata, context: RequestContext
    ) -> None:
        metadata = ThreadMetadata.model_validate(thread.model_dump(exclude={"items"}))
        self._threads[thread.id] = (metadata, datetime.now(UTC))
        self._items.setdefault(thread.id, [])

    async def load_threads(
        self,
        limit: int,
        after: str | None,
        order: str,
        context: RequestContext,
    ) -> Page[ThreadMetadata]:
        threads = [t for t, _ in self._threads.values()]
        threads.sort(key=lambda t: t.created_at, reverse=(order.lower() != "asc"))
        if after:
            ids = [t.id for t in threads]
            if after in ids:
                threads = threads[ids.index(after) + 1 :]
        has_more = len(threads) > limit
        page = threads[:limit]
        return Page(data=page, has_more=has_more, after=page[-1].id if has_more else None)

    async def delete_thread(self, thread_id: str, context: RequestContext) -> None:
        self._threads.pop(thread_id, None)
        self._items.pop(thread_id, None)
        self._runs = {
            wf_id: run
            for wf_id, run in self._runs.items()
            if run.thread_id != thread_id
        }

    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: RequestContext,
    ) -> Page[ThreadItem]:
        items = list(self._items.get(thread_id, []))
        if order.lower() != "asc":
            items = list(reversed(items))
        if after:
            ids = [i.id for i in items]
            if after in ids:
                items = items[ids.index(after) + 1 :]
        has_more = len(items) > limit
        page = items[:limit]
        return Page(data=page, has_more=has_more, after=page[-1].id if has_more else None)

    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: RequestContext
    ) -> None:
        await self.save_item(thread_id, item, context)

    async def save_item(
        self, thread_id: str, item: ThreadItem, context: RequestContext
    ) -> None:
        items = self._items.setdefault(thread_id, [])
        for idx, existing in enumerate(items):
            if existing.id == item.id:
                items[idx] = item
                break
        else:
            items.append(item)
        if thread_id in self._threads:
            thread, _ = self._threads[thread_id]
            self._threads[thread_id] = (thread, datetime.now(UTC))

    async def load_item(
        self, thread_id: str, item_id: str, context: RequestContext
    ) -> ThreadItem:
        for item in self._items.get(thread_id, []):
            if item.id == item_id:
                return item
        raise NotFoundError(f"Item {item_id} not found")

    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: RequestContext
    ) -> None:
        self._items[thread_id] = [
            i for i in self._items.get(thread_id, []) if i.id != item_id
        ]

    async def save_attachment(
        self, attachment: Attachment, context: RequestContext
    ) -> None:
        self._attachments[attachment.id] = attachment

    async def load_attachment(
        self, attachment_id: str, context: RequestContext
    ) -> Attachment:
        if attachment_id not in self._attachments:
            raise NotFoundError(f"Attachment {attachment_id} not found")
        return self._attachments[attachment_id]

    async def delete_attachment(
        self, attachment_id: str, context: RequestContext
    ) -> None:
        self._attachments.pop(attachment_id, None)

    async def create_thread(
        self, title: str | None = None, context: RequestContext | None = None
    ) -> ThreadSummary:
        ctx = context or RequestContext()
        now = datetime.now(UTC)
        thread = ThreadMetadata(
            id=self.generate_thread_id(ctx),
            title=title,
            created_at=now,
        )
        await self.save_thread(thread, ctx)
        return ThreadSummary(
            id=thread.id,
            title=thread.title,
            created_at=now,
            updated_at=now,
            item_count=0,
            latest_phase=None,
        )

    async def list_thread_summaries(self, limit: int = 50) -> list[ThreadSummary]:
        summaries: list[ThreadSummary] = []
        for thread, updated_at in self._threads.values():
            runs = [r for r in self._runs.values() if r.thread_id == thread.id]
            runs.sort(key=lambda r: r.updated_at, reverse=True)
            summaries.append(
                ThreadSummary(
                    id=thread.id,
                    title=thread.title,
                    created_at=thread.created_at,
                    updated_at=updated_at,
                    item_count=len(self._items.get(thread.id, [])),
                    latest_phase=runs[0].phase if runs else None,
                )
            )
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries[:limit]

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
    ) -> ResearchRunRecord:
        now = datetime.now(UTC)
        existing = self._runs.get(workflow_id)
        record = ResearchRunRecord(
            id=existing.id if existing else f"run_{uuid.uuid4().hex[:10]}",
            thread_id=thread_id,
            workflow_id=workflow_id,
            query=query,
            phase=phase,
            message=message,
            sub_questions=sub_questions or [],
            report=report if report is not None else (existing.report if existing else None),
            error=error,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self._runs[workflow_id] = record
        return record

    async def list_research_runs(self, thread_id: str) -> list[ResearchRunRecord]:
        runs = [r for r in self._runs.values() if r.thread_id == thread_id]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs

    async def list_active_research_runs(self) -> list[ResearchRunRecord]:
        runs = [r for r in self._runs.values() if not r.phase.is_terminal]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs


class FakeWorkflowHandle:
    def __init__(self, workflow_id: str, query: str) -> None:
        self.workflow_id = workflow_id
        self._steps = [
            ResearchProgress(
                workflow_id=workflow_id,
                query=query,
                phase=ResearchPhase.PLANNING,
                message="Planning research sub-questions...",
            ),
            ResearchProgress(
                workflow_id=workflow_id,
                query=query,
                phase=ResearchPhase.RESEARCHING,
                message="Researching sub-questions (2/3 completed)...",
                sub_questions=["Q1", "Q2", "Q3"],
                completed_sub_questions=2,
                total_sub_questions=3,
            ),
            ResearchProgress(
                workflow_id=workflow_id,
                query=query,
                phase=ResearchPhase.SYNTHESIZING,
                message="Synthesizing findings into a structured report...",
                sub_questions=["Q1", "Q2", "Q3"],
                completed_sub_questions=3,
                total_sub_questions=3,
            ),
            ResearchProgress(
                workflow_id=workflow_id,
                query=query,
                phase=ResearchPhase.COMPLETED,
                message="Deep research completed.",
                sub_questions=["Q1", "Q2", "Q3"],
                completed_sub_questions=3,
                total_sub_questions=3,
                report=f"# Deep Research Report: {query}\n\nSynthesized findings.",
            ),
        ]
        self._index = 0

    async def get_progress(self) -> ResearchProgress:
        step = self._steps[min(self._index, len(self._steps) - 1)]
        self._index += 1
        return step


class FakeResearchRunner:
    def __init__(self) -> None:
        self.handles: dict[str, FakeWorkflowHandle] = {}

    async def start(self, workflow_id: str, query: str) -> FakeWorkflowHandle:
        handle = FakeWorkflowHandle(workflow_id=workflow_id, query=query)
        self.handles[workflow_id] = handle
        return handle

    def get_handle(self, workflow_id: str) -> FakeWorkflowHandle:
        return self.handles[workflow_id]


def _parse_sse_events(body: str) -> list[dict]:
    events: list[dict] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.mark.asyncio
async def test_health_threads_crud_and_chatkit_streaming() -> None:
    store = InMemoryChatKitStore()
    runner = FakeResearchRunner()
    app = create_app(
        settings=Settings(openai_api_key=""),
        store=store,
        runner=runner,
        poll_interval_seconds=0.0,
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Health check
            health_res = await client.get("/health")
            assert health_res.status_code == 200
            assert health_res.json() == {
                "status": "ok",
                "postgres": True,
                "temporal": True,
                "openai_mode": "demo",
            }

            # 2. Thread CRUD via /api/threads
            create_res = await client.post("/api/threads", json={"title": "Initial Thread"})
            assert create_res.status_code == 200
            created_thread = create_res.json()
            assert created_thread["title"] == "Initial Thread"
            created_id = created_thread["id"]

            list_res = await client.get("/api/threads")
            assert list_res.status_code == 200
            assert any(t["id"] == created_id for t in list_res.json()["threads"])

            # 3. ChatKit streaming request (threads.create)
            chatkit_payload = {
                "type": "threads.create",
                "params": {
                    "input": {
                        "content": [
                            {"type": "input_text", "text": "Compare Temporal and Celery"}
                        ],
                        "attachments": [],
                        "inference_options": {},
                    }
                },
            }
            stream_res = await client.post("/chatkit", json=chatkit_payload)
            assert stream_res.status_code == 200
            assert "text/event-stream" in stream_res.headers["content-type"]

            events = _parse_sse_events(stream_res.text)
            event_types = [e["type"] for e in events]
            assert "thread.created" in event_types
            assert "progress_update" in event_types
            assert "thread.item.done" in event_types

            progress_texts = [
                e["text"] for e in events if e["type"] == "progress_update"
            ]
            assert any("Planning" in t for t in progress_texts)
            assert any("Researching" in t for t in progress_texts)
            assert any("Synthesizing" in t for t in progress_texts)

            assistant_done_events = [
                e
                for e in events
                if e["type"] == "thread.item.done"
                and e["item"]["type"] == "assistant_message"
            ]
            assert len(assistant_done_events) == 1
            assistant_text = assistant_done_events[0]["item"]["content"][0]["text"]
            assert "# Deep Research Report: Compare Temporal and Celery" in assistant_text

            chatkit_thread_id = assistant_done_events[0]["item"]["thread_id"]

            # 4. Verify thread title auto-set and research run recorded
            threads_after = (await client.get("/api/threads")).json()["threads"]
            chatkit_thread = next(t for t in threads_after if t["id"] == chatkit_thread_id)
            assert chatkit_thread["title"] == "Compare Temporal and Celery"
            assert chatkit_thread["latest_phase"] == "completed"

            runs_res = await client.get(f"/api/threads/{chatkit_thread_id}/runs")
            assert runs_res.status_code == 200
            runs = runs_res.json()["runs"]
            assert len(runs) == 1
            assert runs[0]["phase"] == "completed"
            assert runs[0]["sub_questions"] == ["Q1", "Q2", "Q3"]
            assert "Synthesized findings." in runs[0]["report"]

            # 5. Delete thread and verify 404 on second delete
            del_res = await client.delete(f"/api/threads/{created_id}")
            assert del_res.status_code == 200
            assert del_res.json() == {"deleted": True, "id": created_id}

            del_again = await client.delete(f"/api/threads/{created_id}")
            assert del_again.status_code == 404


class GatedWorkflowHandle:
    def __init__(self, workflow_id: str, query: str) -> None:
        self.workflow_id = workflow_id
        self.query = query
        self.release_completion = asyncio.Event()
        self.planning_observed = asyncio.Event()

    async def get_progress(self) -> ResearchProgress:
        if not self.release_completion.is_set():
            self.planning_observed.set()
            return ResearchProgress(
                workflow_id=self.workflow_id,
                query=self.query,
                phase=ResearchPhase.PLANNING,
                message="Planning research sub-questions...",
            )
        return ResearchProgress(
            workflow_id=self.workflow_id,
            query=self.query,
            phase=ResearchPhase.COMPLETED,
            message="Deep research completed.",
            sub_questions=["Q1", "Q2"],
            completed_sub_questions=2,
            total_sub_questions=2,
            report=f"# Deep Research Report: {self.query}\n\nCompleted in background.",
        )


class GatedResearchRunner:
    def __init__(self) -> None:
        self.handles: dict[str, GatedWorkflowHandle] = {}

    async def start(self, workflow_id: str, query: str) -> GatedWorkflowHandle:
        handle = GatedWorkflowHandle(workflow_id=workflow_id, query=query)
        self.handles[workflow_id] = handle
        return handle

    def get_handle(self, workflow_id: str) -> GatedWorkflowHandle:
        return self.handles[workflow_id]


@pytest.mark.asyncio
async def test_research_persists_report_when_client_disconnects_mid_stream() -> None:
    store = InMemoryChatKitStore()
    runner = GatedResearchRunner()
    app = create_app(
        settings=Settings(openai_api_key=""),
        store=store,
        runner=runner,
        poll_interval_seconds=0.01,
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chatkit_payload = {
                "type": "threads.create",
                "params": {
                    "input": {
                        "content": [
                            {"type": "input_text", "text": "Durable workflow test"}
                        ],
                        "attachments": [],
                        "inference_options": {},
                    }
                },
            }

            # Start streaming directly through ChatKitServer and close the SSE generator
            # mid-stream (simulating closing the browser tab while research is still running).
            created_thread_id: str | None = None
            stream_result = await app.state.chatkit_server.process(
                json.dumps(chatkit_payload).encode("utf-8"),
                RequestContext(),
            )
            stream_iter = stream_result.json_events
            try:
                async for chunk in stream_iter:
                    line = chunk.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: ") :])
                    if event["type"] == "thread.created":
                        created_thread_id = event["thread"]["id"]
                    if event["type"] == "progress_update" and "Planning" in event["text"]:
                        break
            finally:
                await stream_iter.aclose()

            assert created_thread_id is not None
            assert len(runner.handles) == 1
            handle = next(iter(runner.handles.values()))

            # Complete the Temporal workflow AFTER the browser tab closed.
            handle.release_completion.set()
            for _ in range(50):
                runs = await store.list_research_runs(created_thread_id)
                if runs and runs[0].phase == ResearchPhase.COMPLETED:
                    break
                await asyncio.sleep(0.02)

            # Reopen browser tab: /api/threads and threads.get_by_id must return the completed report.
            threads_res = await client.get("/api/threads")
            assert threads_res.status_code == 200
            thread_summary = next(
                t for t in threads_res.json()["threads"] if t["id"] == created_thread_id
            )
            assert thread_summary["latest_phase"] == "completed"

            reopen_res = await client.post(
                "/chatkit",
                json={
                    "type": "threads.get_by_id",
                    "params": {"thread_id": created_thread_id},
                },
            )
            assert reopen_res.status_code == 200
            items = reopen_res.json()["items"]["data"]
            assistant_items = [i for i in items if i["type"] == "assistant_message"]
            assert len(assistant_items) == 1
            assert "Completed in background." in assistant_items[0]["content"][0]["text"]


@pytest.mark.asyncio
async def test_reconcile_persists_completed_run_after_server_restart() -> None:
    store = InMemoryChatKitStore()
    runner = GatedResearchRunner()
    app = create_app(
        settings=Settings(openai_api_key=""),
        store=store,
        runner=runner,
        poll_interval_seconds=0.01,
    )

    async with app.router.lifespan_context(app):
        created = await store.create_thread(title="Orphaned run")
        workflow_id = f"deep-research-{created.id}-reconcile"
        handle = await runner.start(workflow_id=workflow_id, query="Orphaned query")
        handle.release_completion.set()

        # Simulate DB row left in 'researching' before a server restart.
        await store.upsert_research_run(
            thread_id=created.id,
            workflow_id=workflow_id,
            query="Orphaned query",
            phase=ResearchPhase.RESEARCHING,
            message="Researching sub-questions...",
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            reopen_res = await client.post(
                "/chatkit",
                json={
                    "type": "threads.get_by_id",
                    "params": {"thread_id": created.id},
                },
            )
            assert reopen_res.status_code == 200
            items = reopen_res.json()["items"]["data"]
            assistant_items = [i for i in items if i["type"] == "assistant_message"]
            assert len(assistant_items) == 1
            assert "Completed in background." in assistant_items[0]["content"][0]["text"]

            runs = await store.list_research_runs(created.id)
            assert runs[0].phase == ResearchPhase.COMPLETED


@pytest.mark.asyncio
async def test_resume_custom_action_reattaches_live_native_stream() -> None:
    store = InMemoryChatKitStore()
    runner = GatedResearchRunner()
    app = create_app(
        settings=Settings(openai_api_key=""),
        store=store,
        runner=runner,
        poll_interval_seconds=0.01,
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chatkit_payload = {
                "type": "threads.create",
                "params": {
                    "input": {
                        "content": [
                            {"type": "input_text", "text": "Resume live stream test"}
                        ],
                        "attachments": [],
                        "inference_options": {},
                    }
                },
            }

            # 1. Start stream and disconnect mid-flight (simulating closing the tab).
            created_thread_id: str | None = None
            stream_result = await app.state.chatkit_server.process(
                json.dumps(chatkit_payload).encode("utf-8"),
                RequestContext(),
            )
            stream_iter = stream_result.json_events
            try:
                async for chunk in stream_iter:
                    line = chunk.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: ") :])
                    if event["type"] == "thread.created":
                        created_thread_id = event["thread"]["id"]
                    if event["type"] == "progress_update" and "Planning" in event["text"]:
                        break
            finally:
                await stream_iter.aclose()

            assert created_thread_id is not None
            handle = next(iter(runner.handles.values()))

            # 2. Reopen tab and re-attach via threads.custom_action (no item_id).
            resume_payload = {
                "type": "threads.custom_action",
                "params": {
                    "thread_id": created_thread_id,
                    "action": {
                        "type": "research.resume",
                        "payload": {"thread_id": created_thread_id},
                    },
                },
            }

            async def _complete_soon() -> None:
                await asyncio.sleep(0.03)
                handle.release_completion.set()

            completer = asyncio.create_task(_complete_soon())
            try:
                resume_res = await client.post("/chatkit", json=resume_payload)
            finally:
                await completer

            assert resume_res.status_code == 200
            assert "text/event-stream" in resume_res.headers["content-type"]

            events = _parse_sse_events(resume_res.text)
            stream_opt_events = [e for e in events if e["type"] == "stream_options"]
            assert stream_opt_events
            assert stream_opt_events[0]["stream_options"]["allow_cancel"] is False

            progress_texts = [
                e["text"] for e in events if e["type"] == "progress_update"
            ]
            assert any("Planning" in t for t in progress_texts)
            assert any("completed" in t.lower() for t in progress_texts)

            done_items = [
                e["item"]
                for e in events
                if e["type"] == "thread.item.done"
                and e["item"]["type"] == "assistant_message"
            ]
            assert len(done_items) == 1
            assert "Completed in background." in done_items[0]["content"][0]["text"]

            # 3. Verify no duplicate assistant messages were persisted.
            thread_res = await client.post(
                "/chatkit",
                json={
                    "type": "threads.get_by_id",
                    "params": {"thread_id": created_thread_id},
                },
            )
            assistant_items = [
                i
                for i in thread_res.json()["items"]["data"]
                if i["type"] == "assistant_message"
            ]
            assert len(assistant_items) == 1


@pytest.mark.asyncio
async def test_resume_custom_action_reconciles_after_restart_and_handles_terminal_race() -> None:
    store = InMemoryChatKitStore()
    runner = GatedResearchRunner()
    app = create_app(
        settings=Settings(openai_api_key=""),
        store=store,
        runner=runner,
        poll_interval_seconds=0.01,
    )

    async with app.router.lifespan_context(app):
        created = await store.create_thread(title="Restarted stream")
        workflow_id = f"deep-research-{created.id}-restart"
        handle = await runner.start(workflow_id=workflow_id, query="Restart query")

        # Simulate DB row left in 'planning' after a server restart (no active monitor yet).
        await store.upsert_research_run(
            thread_id=created.id,
            workflow_id=workflow_id,
            query="Restart query",
            phase=ResearchPhase.PLANNING,
            message="Planning research sub-questions...",
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resume_payload = {
                "type": "threads.custom_action",
                "params": {
                    "thread_id": created.id,
                    "action": {
                        "type": "research.resume",
                        "payload": {"workflow_id": workflow_id},
                    },
                },
            }

            async def _complete_soon() -> None:
                await asyncio.sleep(0.03)
                handle.release_completion.set()

            completer = asyncio.create_task(_complete_soon())
            try:
                resume_res = await client.post("/chatkit", json=resume_payload)
            finally:
                await completer

            assert resume_res.status_code == 200
            events = _parse_sse_events(resume_res.text)
            done_items = [
                e["item"]
                for e in events
                if e["type"] == "thread.item.done"
                and e["item"]["type"] == "assistant_message"
            ]
            assert len(done_items) == 1
            assert "Completed in background." in done_items[0]["content"][0]["text"]

            # Calling research.resume again after completion (race window) returns the
            # persisted item idempotently without duplicating items in the store.
            race_res = await client.post("/chatkit", json=resume_payload)
            assert race_res.status_code == 200
            race_events = _parse_sse_events(race_res.text)
            race_done = [
                e["item"]
                for e in race_events
                if e["type"] == "thread.item.done"
                and e["item"]["type"] == "assistant_message"
            ]
            assert len(race_done) == 1
            assert race_done[0]["id"] == done_items[0]["id"]

            items = (await store.load_thread_items(created.id, None, 20, "asc", RequestContext())).data
            assert len([i for i in items if i.type == "assistant_message"]) == 1


class ApprovalWorkflowHandle:
    def __init__(self, workflow_id: str, query: str) -> None:
        self.workflow_id = workflow_id
        self.query = query
        self.decision: ApprovalDecision | None = None
        self._post_approval_step = 0

    async def submit_approval(self, decision: ApprovalDecision) -> None:
        self.decision = decision

    async def get_progress(self) -> ResearchProgress:
        if self.decision is None:
            return ResearchProgress(
                workflow_id=self.workflow_id,
                query=self.query,
                phase=ResearchPhase.AWAITING_APPROVAL,
                message="Waiting for your approval on the research plan...",
                sub_questions=["Architecture tradeoffs", "Operational cost"],
                total_sub_questions=2,
            )

        selected = list(self.decision.sub_questions)
        if self.decision.extra_focus:
            selected.append(self.decision.extra_focus)

        findings = [
            ResearchFinding(
                sub_question=q,
                summary=f"Summary for {q}",
                key_points=[f"Key insight about {q}"],
                sources=[f"https://example.com/{idx}"],
            )
            for idx, q in enumerate(selected)
        ]

        if self._post_approval_step == 0:
            self._post_approval_step += 1
            return ResearchProgress(
                workflow_id=self.workflow_id,
                query=self.query,
                phase=ResearchPhase.RESEARCHING,
                message=f"Researching {len(selected)} sub-questions...",
                sub_questions=selected,
                completed_sub_questions=1,
                total_sub_questions=len(selected),
            )

        return ResearchProgress(
            workflow_id=self.workflow_id,
            query=self.query,
            phase=ResearchPhase.COMPLETED,
            message="Deep research completed.",
            sub_questions=selected,
            completed_sub_questions=len(selected),
            total_sub_questions=len(selected),
            findings=findings,
            approval_decision=self.decision,
            report=f"# Deep Research Report: {self.query}\n\nApproved findings.",
        )


class ApprovalResearchRunner:
    def __init__(self) -> None:
        self.handles: dict[str, ApprovalWorkflowHandle] = {}

    async def start(self, workflow_id: str, query: str) -> ApprovalWorkflowHandle:
        handle = ApprovalWorkflowHandle(workflow_id=workflow_id, query=query)
        self.handles[workflow_id] = handle
        return handle

    def get_handle(self, workflow_id: str) -> ApprovalWorkflowHandle:
        return self.handles[workflow_id]


@pytest.mark.asyncio
async def test_hitl_approval_and_all_four_chatkit_widgets() -> None:
    store = InMemoryChatKitStore()
    runner = ApprovalResearchRunner()
    app = create_app(
        settings=Settings(openai_api_key=""),
        store=store,
        runner=runner,
        poll_interval_seconds=0.01,
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. User starts research -> receives Widget 1 (HITL Approval Card) and stream pauses
            create_res = await client.post(
                "/chatkit",
                json={
                    "type": "threads.create",
                    "params": {
                        "input": {
                            "content": [
                                {"type": "input_text", "text": "Evaluate vector databases"}
                            ],
                            "attachments": [],
                            "inference_options": {},
                        }
                    },
                },
            )
            assert create_res.status_code == 200
            events = _parse_sse_events(create_res.text)
            thread_id = next(e["thread"]["id"] for e in events if e["type"] == "thread.created")
            approval_widgets = [
                e["item"]
                for e in events
                if e["type"] == "thread.item.done"
                and e["item"]["type"] == "widget"
                and e["item"]["id"].startswith("wgt_approval_")
            ]
            assert len(approval_widgets) == 1
            approval_widget = approval_widgets[0]
            user_item = next(
                event["item"]
                for event in events
                if event["type"] == "thread.item.done"
                and event["item"]["type"] == "user_message"
            )
            prompt_time = datetime.fromisoformat(user_item["created_at"])
            approval_time = datetime.fromisoformat(approval_widget["created_at"])
            assert prompt_time.tzinfo is not None
            assert prompt_time < approval_time
            assert approval_widget["widget"]["type"] == "Card"
            assert approval_widget["widget"]["asForm"] is True

            workflow_id = next(iter(runner.handles))

            # 2. Human approves plan -> Widget 1 updates in place, Widgets 2, 3, 4 & report stream
            approve_res = await client.post(
                "/chatkit",
                json={
                    "type": "threads.custom_action",
                    "params": {
                        "thread_id": thread_id,
                        "item_id": approval_widget["id"],
                        "action": {
                            "type": "research.approve_plan",
                            "payload": {
                                "workflow_id": workflow_id,
                                "sub_questions": [
                                    "Architecture tradeoffs",
                                    "Operational cost",
                                ],
                                "q_0": True,
                                "q_1": False,
                                "depth": "deep",
                                "extra_focus": "Latency benchmarks",
                            },
                        },
                    },
                },
            )
            assert approve_res.status_code == 200
            approve_events = _parse_sse_events(approve_res.text)

            replaced_ids = [
                e["item"]["id"]
                for e in approve_events
                if e["type"] == "thread.item.replaced"
            ]
            done_widget_ids = [
                e["item"]["id"]
                for e in approve_events
                if e["type"] == "thread.item.done" and e["item"]["type"] == "widget"
            ]
            assert approval_widget["id"] in replaced_ids
            assert f"wgt_progress_{workflow_id}" in done_widget_ids
            assert f"wgt_findings_{workflow_id}" in done_widget_ids
            assert f"wgt_actions_{workflow_id}" in done_widget_ids

            # Verify human selection was signalled to the workflow
            handle = runner.handles[workflow_id]
            assert handle.decision is not None
            assert handle.decision.approved is True
            assert handle.decision.sub_questions == ["Architecture tradeoffs"]
            assert handle.decision.extra_focus == "Latency benchmarks"

            # 3. Click finding in Widget 3 -> sync_action expands it in place
            sync_res = await client.post(
                "/chatkit",
                json={
                    "type": "threads.sync_custom_action",
                    "params": {
                        "thread_id": thread_id,
                        "item_id": f"wgt_findings_{workflow_id}",
                        "action": {
                            "type": "research.toggle_finding",
                            "payload": {
                                "workflow_id": workflow_id,
                                "expanded_indices": [0],
                            },
                        },
                    },
                },
            )
            assert sync_res.status_code == 200
            sync_body = sync_res.json()
            assert sync_body["updated_item"]["id"] == f"wgt_findings_{workflow_id}"
            assert "Key insight about Architecture tradeoffs" in json.dumps(sync_body)

            # 4. Click TL;DR and Sources buttons in Widget 4
            tldr_res = await client.post(
                "/chatkit",
                json={
                    "type": "threads.custom_action",
                    "params": {
                        "thread_id": thread_id,
                        "item_id": f"wgt_actions_{workflow_id}",
                        "action": {
                            "type": "research.followup",
                            "payload": {
                                "workflow_id": workflow_id,
                                "query": "Evaluate vector databases",
                                "mode": "tldr",
                            },
                        },
                    },
                },
            )
            assert tldr_res.status_code == 200
            assert "Executive TL;DR" in tldr_res.text

            sources_res = await client.post(
                "/chatkit",
                json={
                    "type": "threads.custom_action",
                    "params": {
                        "thread_id": thread_id,
                        "item_id": f"wgt_actions_{workflow_id}",
                        "action": {
                            "type": "research.followup",
                            "payload": {
                                "workflow_id": workflow_id,
                                "query": "Evaluate vector databases",
                                "mode": "sources",
                            },
                        },
                    },
                },
            )
            assert sources_res.status_code == 200
            assert "https://example.com/0" in sources_res.text
