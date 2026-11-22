from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import httpx
import pytest
from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata

from app.config import Settings
from app.domain import (
    RequestContext,
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
        self._threads[thread.id] = (thread.model_copy(deep=True), datetime.now(UTC))
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
    async def start(self, workflow_id: str, query: str) -> FakeWorkflowHandle:
        return FakeWorkflowHandle(workflow_id=workflow_id, query=query)


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
