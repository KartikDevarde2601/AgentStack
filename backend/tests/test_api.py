from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import httpx
import pytest
from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin
from temporalio.contrib.openai_agents.testing import TestModelProvider
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.config import Settings
from app.domain import (
    RequestContext,
    ResearchPhase,
    ResearchRunRecord,
    ThreadSummary,
)
from app.events import InMemoryThreadEventBus
from app.main import create_app
from app.temporal.activities import ThreadActivities
from app.temporal.client import TemporalClientManager
from app.temporal.orchestrator import TemporalThreadOrchestrator
from app.temporal.workflows import AgentThreadWorkflow
from tests.fake_agents_model import RoutingFakeModel


class InMemoryChatKitStore(Store[RequestContext]):
    """In-memory implementation of PostgresChatKitStore for API tests."""

    def __init__(self) -> None:
        self._threads: dict[str, tuple[ThreadMetadata, datetime]] = {}
        self._conversations: dict[str, str] = {}
        self._items: dict[str, list[ThreadItem]] = {}
        self._attachments: dict[str, Attachment] = {}
        self._runs: dict[str, ResearchRunRecord] = {}

    async def ping(self) -> bool:
        return True

    async def get_openai_conversation_id(self, thread_id: str) -> str | None:
        return self._conversations.get(thread_id)

    async def set_openai_conversation_id(
        self, thread_id: str, conversation_id: str
    ) -> None:
        self._conversations[thread_id] = conversation_id

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
        self._conversations.pop(thread_id, None)
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


class FakeConversationClient:
    async def create_conversation(self) -> str:
        return f"conv_{uuid.uuid4().hex[:8]}"

    async def append_conversation_messages(
        self, conversation_id: str, messages: list[dict[str, str]]
    ) -> None:
        return None


def _parse_sse_events(body: str) -> list[dict]:
    events: list[dict] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def _worker_activities(activities: ThreadActivities) -> list:
    return [
        activities.ensure_openai_conversation_activity,
        activities.record_failed_turns_activity,
        activities.persist_user_message_activity,
        activities.publish_progress_activity,
        activities.persist_clarifications_widget_activity,
        activities.persist_approval_widget_activity,
        activities.publish_report_activity,
        activities.persist_final_item_activity,
        activities.publish_turn_end_activity,
        activities.prune_thread_events_activity,
    ]


@pytest.mark.asyncio
async def test_health_threads_crud_clarifications_approval_and_streaming() -> None:
    store = InMemoryChatKitStore()
    bus = InMemoryThreadEventBus()
    conv_client = FakeConversationClient()
    fake_model = RoutingFakeModel(clarify_keywords=("best",))
    activities = ThreadActivities(client=conv_client, store=store, event_bus=bus)

    plugin = OpenAIAgentsPlugin(model_provider=TestModelProvider(fake_model))
    async with await WorkflowEnvironment.start_time_skipping(plugins=[plugin]) as env:
        task_queue = f"test-api-{uuid.uuid4().hex[:8]}"
        client_manager = TemporalClientManager.from_client(env.client)
        orchestrator = TemporalThreadOrchestrator(
            client_manager, task_queue=task_queue, require_approval=True
        )
        app = create_app(
            settings=Settings(openai_api_key="sk-test"),
            store=store,
            orchestrator=orchestrator,
            event_bus=bus,
            temporal_client_manager=client_manager,
        )

        async with (
            Worker(
                env.client,
                task_queue=task_queue,
                workflows=[AgentThreadWorkflow],
                activities=_worker_activities(activities),
            ),
            app.router.lifespan_context(app),
        ):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # 1. Health endpoint
                health_res = await client.get("/health")
                assert health_res.status_code == 200
                assert health_res.json() == {
                    "status": "ok",
                    "postgres": True,
                    "temporal": True,
                    "openai_mode": "live",
                }

                # 2. Thread CRUD
                create_res = await client.post(
                    "/api/threads", json={"title": "Initial Thread"}
                )
                assert create_res.status_code == 200
                created_id = create_res.json()["id"]
                del_res = await client.delete(f"/api/threads/{created_id}")
                assert del_res.status_code == 200
                assert (await client.delete(f"/api/threads/{created_id}")).status_code == 404

                # 3. Broad query pauses at Clarifications Gate with dynamic widget
                stream_res = await client.post(
                    "/chatkit",
                    json={
                        "type": "threads.create",
                        "params": {
                            "input": {
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": "Best agent frameworks",
                                    }
                                ],
                                "attachments": [],
                                "inference_options": {},
                            }
                        },
                    },
                )
                assert stream_res.status_code == 200
                events = _parse_sse_events(stream_res.text)
                created_ev = next(e for e in events if e["type"] == "thread.created")
                thread_id = created_ev["thread"]["id"]
                widget_events = [
                    e
                    for e in events
                    if e["type"] == "thread.item.done"
                    and e["item"]["type"] == "widget"
                ]
                assert len(widget_events) == 1

                state_res = await client.get(f"/api/threads/{thread_id}/state")
                state_json = state_res.json()
                assert state_json["phase"] == "awaiting_clarifications"
                assert state_json["pending_clarification"] is not None

                # Sending another message while awaiting clarifications reminds user
                blocked_clarify = await client.post(
                    "/chatkit",
                    json={
                        "type": "threads.add_user_message",
                        "params": {
                            "thread_id": thread_id,
                            "input": {
                                "content": [{"type": "input_text", "text": "hello"}],
                                "attachments": [],
                                "inference_options": {},
                            },
                        },
                    },
                )
                assert blocked_clarify.status_code == 200
                assert any(
                    e["type"] == "thread.item.done"
                    and e["item"]["type"] == "assistant_message"
                    and "clarifying questions" in e["item"]["content"][0]["text"].lower()
                    for e in _parse_sse_events(blocked_clarify.text)
                )

                # Stale clarification card click is rejected cleanly via @workflow.update validator
                stale_clarify_res = await client.post(
                    "/chatkit",
                    json={
                        "type": "threads.custom_action",
                        "params": {
                            "thread_id": thread_id,
                            "action": {
                                "type": "research.submit_clarifications",
                                "payload": {
                                    "workflow_id": "turn_stale_000",
                                    "questions": state_json["pending_clarification"][
                                        "questions"
                                    ],
                                    "q_0": "production",
                                },
                            },
                        },
                    },
                )
                assert stale_clarify_res.status_code == 200
                assert any(
                    e["type"] == "thread.item.done"
                    and e["item"]["type"] == "assistant_message"
                    and "earlier turn" in e["item"]["content"][0]["text"].lower()
                    for e in _parse_sse_events(stale_clarify_res.text)
                )

                # 4. Submit dynamic clarifications -> pauses at Plan Approval Gate
                clarify_res = await client.post(
                    "/chatkit",
                    json={
                        "type": "threads.custom_action",
                        "params": {
                            "thread_id": thread_id,
                            "action": {
                                "type": "research.submit_clarifications",
                                "payload": {
                                    "workflow_id": state_json["active_turn_id"],
                                    "questions": state_json["pending_clarification"][
                                        "questions"
                                    ],
                                    "q_0": "production",
                                    "q_1__opt__0": True,
                                    "q_1__opt__2": True,
                                    "q_2": "Python + Temporal stack",
                                },
                            },
                        },
                    },
                )
                assert clarify_res.status_code == 200

                state_json = (await client.get(f"/api/threads/{thread_id}/state")).json()
                assert state_json["phase"] == "awaiting_approval"
                assert state_json["pending_approval"] is not None

                # 5. Approve plan -> streams deltas & final report
                approve_res = await client.post(
                    "/chatkit",
                    json={
                        "type": "threads.custom_action",
                        "params": {
                            "thread_id": thread_id,
                            "action": {
                                "type": "research.approve_plan",
                                "payload": {
                                    "workflow_id": state_json["active_turn_id"],
                                    "sub_questions": state_json["pending_approval"][
                                        "sub_questions"
                                    ],
                                    "q_0": True,
                                    "q_1": True,
                                    "depth": "deep",
                                },
                            },
                        },
                    },
                )
                assert approve_res.status_code == 200
                approve_events = _parse_sse_events(approve_res.text)
                assert any(
                    e["type"] == "thread.item.updated" for e in approve_events
                )
                assistant_done = [
                    e
                    for e in approve_events
                    if e["type"] == "thread.item.done"
                    and e["item"]["type"] == "assistant_message"
                ]
                assert len(assistant_done) == 1
                assert (
                    "# Deep Research Report:"
                    in assistant_done[0]["item"]["content"][0]["text"]
                )


@pytest.mark.asyncio
async def test_streaming_report_opens_and_closes_chatkit_content_part() -> None:
    store = InMemoryChatKitStore()
    bus = InMemoryThreadEventBus()
    conv_client = FakeConversationClient()
    fake_model = RoutingFakeModel()
    activities = ThreadActivities(client=conv_client, store=store, event_bus=bus)

    plugin = OpenAIAgentsPlugin(model_provider=TestModelProvider(fake_model))
    async with await WorkflowEnvironment.start_time_skipping(plugins=[plugin]) as env:
        task_queue = f"test-stream-proto-{uuid.uuid4().hex[:8]}"
        client_manager = TemporalClientManager.from_client(env.client)
        orchestrator = TemporalThreadOrchestrator(
            client_manager, task_queue=task_queue, require_approval=False
        )
        app = create_app(
            settings=Settings(openai_api_key="sk-test"),
            store=store,
            orchestrator=orchestrator,
            event_bus=bus,
            temporal_client_manager=client_manager,
        )

        async with (
            Worker(
                env.client,
                task_queue=task_queue,
                workflows=[AgentThreadWorkflow],
                activities=_worker_activities(activities),
            ),
            app.router.lifespan_context(app),
        ):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                res = await client.post(
                    "/chatkit",
                    json={
                        "type": "threads.create",
                        "params": {
                            "input": {
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": "Compare Temporal and Celery",
                                    }
                                ],
                                "attachments": [],
                                "inference_options": {},
                            }
                        },
                    },
                )
                assert res.status_code == 200
                events = _parse_sse_events(res.text)

    added_idx, added_ev = next(
        (i, e)
        for i, e in enumerate(events)
        if e["type"] == "thread.item.added"
        and e["item"]["type"] == "assistant_message"
    )
    assistant_id = added_ev["item"]["id"]
    assert added_ev["item"]["content"] == []

    msg_updates = [
        (i, e["update"]["type"])
        for i, e in enumerate(events)
        if e["type"] == "thread.item.updated" and e["item_id"] == assistant_id
    ]
    assert msg_updates, "expected streaming updates for assistant message"
    assert msg_updates[0][1] == "assistant_message.content_part.added"
    assert msg_updates[0][0] > added_idx
    assert any(kind == "assistant_message.content_part.text_delta" for _, kind in msg_updates)
    assert msg_updates[-1][1] == "assistant_message.content_part.done"

    done_idx = next(
        i
        for i, e in enumerate(events)
        if e["type"] == "thread.item.done" and e["item"]["id"] == assistant_id
    )
    assert msg_updates[-1][0] < done_idx
