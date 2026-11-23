from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest
from chatkit.types import AssistantMessageItem, ThreadItem
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.domain import (
    ApprovalDecision,
    IncomingMessage,
    RequestContext,
    ResearchFinding,
    ResearchPhase,
    ResearchPlan,
    ResearchRunRecord,
)
from app.events import InMemoryThreadEventBus, ThreadEventKind
from app.temporal.activities import ThreadActivities
from app.temporal.orchestrator import TemporalThreadOrchestrator, thread_workflow_id
from app.temporal.workflows import AgentThreadInput, AgentThreadWorkflow


class FakeThreadStore:
    def __init__(self) -> None:
        self.conversations: dict[str, str] = {}
        self.items: dict[str, list[ThreadItem]] = {}
        self.runs: dict[str, ResearchRunRecord] = {}

    async def get_openai_conversation_id(self, thread_id: str) -> str | None:
        return self.conversations.get(thread_id)

    async def set_openai_conversation_id(
        self, thread_id: str, conversation_id: str
    ) -> None:
        self.conversations[thread_id] = conversation_id

    async def save_item(
        self, thread_id: str, item: ThreadItem, context: RequestContext
    ) -> None:
        bucket = self.items.setdefault(thread_id, [])
        for idx, existing in enumerate(bucket):
            if existing.id == item.id:
                bucket[idx] = item
                return
        bucket.append(item)

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
        record = ResearchRunRecord(
            id=f"run_{workflow_id}",
            thread_id=thread_id,
            workflow_id=workflow_id,
            query=query,
            phase=phase,
            message=message,
            sub_questions=sub_questions or [],
            report=report,
            error=error,
            created_at=now,
            updated_at=now,
        )
        self.runs[workflow_id] = record
        return record


class FakeResearchLLMClient:
    def __init__(self) -> None:
        self.created_conversations: list[str] = []
        self.investigated: list[tuple[str, str, str]] = []

    async def create_conversation(self) -> str:
        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        self.created_conversations.append(conv_id)
        return conv_id

    async def append_conversation_messages(
        self, conversation_id: str, messages: list[dict[str, str]]
    ) -> None:
        return None

    async def generate_research_plan(
        self, query: str, *, conversation_id: str | None = None
    ) -> ResearchPlan:
        return ResearchPlan(
            query=query,
            sub_questions=[
                f"Sub-question 1 for {query}",
                f"Sub-question 2 for {query}",
            ],
        )

    async def investigate_sub_question(
        self,
        query: str,
        sub_question: str,
        *,
        depth: str = "standard",
        conversation_id: str | None = None,
    ) -> ResearchFinding:
        self.investigated.append((query, sub_question, depth))
        return ResearchFinding(
            sub_question=sub_question,
            summary=f"Summary of {sub_question}",
            key_points=[f"Point A ({depth})"],
            sources=["https://example.com/source"],
        )

    async def synthesize_report_stream(
        self,
        query: str,
        findings: list[ResearchFinding],
        *,
        conversation_id: str | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        parts = [f"# Report: {query}\n\n", f"Synthesized {len(findings)} findings."]
        for chunk in parts:
            if on_delta is not None:
                await on_delta(chunk)
        return "".join(parts)


def _all_activities(activities: ThreadActivities) -> list:
    return [
        activities.ensure_openai_conversation_activity,
        activities.record_failed_turns_activity,
        activities.persist_user_message_activity,
        activities.publish_progress_activity,
        activities.persist_approval_widget_activity,
        activities.plan_research_activity,
        activities.investigate_sub_question_activity,
        activities.synthesize_report_activity,
        activities.persist_final_item_activity,
        activities.publish_turn_end_activity,
        activities.prune_thread_events_activity,
    ]


@pytest.mark.asyncio
async def test_agent_thread_workflow_approval_and_streaming() -> None:
    store = FakeThreadStore()
    bus = InMemoryThreadEventBus()
    llm = FakeResearchLLMClient()
    activities = ThreadActivities(client=llm, store=store, event_bus=bus)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        task_queue = f"test-entity-{uuid.uuid4().hex[:8]}"
        orchestrator = TemporalThreadOrchestrator(
            env.client, task_queue=task_queue, require_approval=True
        )
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=_all_activities(activities),
        ):
            thread_id = f"thr_{uuid.uuid4().hex[:8]}"
            await orchestrator.send_message(
                thread_id,
                IncomingMessage(
                    message_id="msg_1",
                    text="Compare Postgres and SQLite",
                    already_persisted=True,
                ),
            )

            # Wait until Turn 1 pauses at the Approval Gate.
            state = await orchestrator.get_state(thread_id)
            for _ in range(100):
                if state.phase == ResearchPhase.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.02)
                state = await orchestrator.get_state(thread_id)

            assert state.phase == ResearchPhase.AWAITING_APPROVAL
            assert state.pending_approval is not None
            assert state.openai_conversation_id is not None

            # Approve Turn 1 with edited sub-questions + extra focus.
            await orchestrator.submit_approval(
                thread_id,
                ApprovalDecision(
                    approved=True,
                    sub_questions=state.pending_approval.sub_questions[:1],
                    depth="deep",
                    extra_focus="WAL mode tradeoffs",
                ),
            )

            # Wait for Turn 1 to finish and workflow to return to IDLE.
            for _ in range(100):
                state = await orchestrator.get_state(thread_id)
                if state.phase == ResearchPhase.IDLE:
                    break
                await asyncio.sleep(0.02)

            assert state.phase == ResearchPhase.IDLE
            assert state.turns_since_continue == 1

            # Verify live streaming deltas were emitted to ThreadEventBus.
            events = [ev for ev in bus._events_by_thread[thread_id]]
            assert any(ev.kind == ThreadEventKind.APPROVAL_PAUSE for ev in events)
            assert any(ev.kind == ThreadEventKind.TEXT_DELTA for ev in events)
            assert any(ev.kind == ThreadEventKind.TURN_END for ev in events)

            # Verify findings and final report were persisted by activities.
            items = store.items[thread_id]
            assistant_items = [i for i in items if isinstance(i, AssistantMessageItem)]
            assert len(assistant_items) == 1
            assert "# Report: Compare Postgres and SQLite" in assistant_items[0].content[0].text
            assert any(sub == "WAL mode tradeoffs" for _, sub, _ in llm.investigated)


@pytest.mark.asyncio
async def test_agent_thread_workflow_reject_and_idle_prune() -> None:
    store = FakeThreadStore()
    bus = InMemoryThreadEventBus()
    llm = FakeResearchLLMClient()
    activities = ThreadActivities(client=llm, store=store, event_bus=bus)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        task_queue = f"test-idle-{uuid.uuid4().hex[:8]}"
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=_all_activities(activities),
        ):
            thread_id = f"thr_{uuid.uuid4().hex[:8]}"
            wf_id = thread_workflow_id(thread_id)
            handle = await env.client.start_workflow(
                AgentThreadWorkflow.run,
                AgentThreadInput(
                    thread_id=thread_id,
                    require_approval=True,
                    idle_timeout_seconds=1800,
                ),
                id=wf_id,
                task_queue=task_queue,
                start_signal="submit_message",
                start_signal_args=[
                    IncomingMessage(
                        message_id="msg_1",
                        text="Turn 1",
                        already_persisted=True,
                    )
                ],
            )

            for _ in range(100):
                state = await handle.query(AgentThreadWorkflow.get_state)
                if state.phase == ResearchPhase.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.02)
            assert state.phase == ResearchPhase.AWAITING_APPROVAL

            await handle.signal(
                AgentThreadWorkflow.submit_approval,
                ApprovalDecision(approved=False, reason="Cancelled by user"),
            )

            # Workflow returns to IDLE and exits after 30m (time-skipped), pruning run_events.
            await handle.result()

            items = store.items[thread_id]
            cancelled_items = [
                i
                for i in items
                if isinstance(i, AssistantMessageItem)
                and "Research Cancelled" in i.content[0].text
            ]
            assert len(cancelled_items) == 1
            assert await bus.latest_seq(thread_id) == 0


@pytest.mark.asyncio
async def test_retry_after_failed_turn_preserves_original_topic_in_conversation() -> None:
    """Regression test: OpenAI responses.create(conversation=...) is transactional,
    so a failed Turn 1 never appends the user's prompt to the Conversation.
    AgentThreadWorkflow must record/flush failed Turns so 'try again' resolves the original topic.
    """

    class TransactionalConversationLLM(FakeResearchLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.conversation_history: dict[str, list[str]] = {}
            self._remaining_plan_failures = 3  # exhausts maximum_attempts=3
            self._remaining_flush_failures = 1  # fails best-effort flush during Turn 1 outage

        async def create_conversation(self) -> str:
            cid = await super().create_conversation()
            self.conversation_history[cid] = []
            return cid

        async def append_conversation_messages(
            self, conversation_id: str, messages: list[dict[str, str]]
        ) -> None:
            if self._remaining_flush_failures > 0:
                self._remaining_flush_failures -= 1
                raise RuntimeError("503 Service Unavailable")
            for msg in messages:
                self.conversation_history.setdefault(conversation_id, []).append(
                    msg["content"]
                )

        async def generate_research_plan(
            self, query: str, *, conversation_id: str | None = None
        ) -> ResearchPlan:
            if self._remaining_plan_failures > 0:
                self._remaining_plan_failures -= 1
                raise RuntimeError("400 Bad Request")

            history = self.conversation_history.get(conversation_id or "", [])
            combined = " ".join(history + [query]).lower()
            if "temporal" not in combined:
                return ResearchPlan(
                    query=query,
                    sub_questions=["What topic or question should the research address?"],
                )
            return ResearchPlan(
                query="Temporal durable execution for deep research agents",
                sub_questions=[
                    "How does Temporal event sourcing recover agent state?",
                    "How do Temporal signals enable HITL approval gates?",
                ],
            )

    store = FakeThreadStore()
    bus = InMemoryThreadEventBus()
    llm = TransactionalConversationLLM()
    activities = ThreadActivities(client=llm, store=store, event_bus=bus)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        task_queue = f"test-retry-{uuid.uuid4().hex[:8]}"
        orchestrator = TemporalThreadOrchestrator(
            env.client, task_queue=task_queue, require_approval=True
        )
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=_all_activities(activities),
        ):
            thread_id = f"thr_{uuid.uuid4().hex[:8]}"

            # Turn 1 fails completely (including best-effort flush during outage).
            await orchestrator.send_message(
                thread_id,
                IncomingMessage(
                    message_id="msg_1",
                    text="Explain how Temporal durable execution improves reliability for multi-step deep research agents.",
                    already_persisted=True,
                ),
            )
            for _ in range(250):
                st = await orchestrator.get_state(thread_id)
                if st.phase == ResearchPhase.IDLE and st.turns_since_continue == 1:
                    break
                await asyncio.sleep(0.02)
            assert st.turns_since_continue == 1

            # Turn 2 ("try again") flushes Turn 1 into the OpenAI Conversation first.
            await orchestrator.send_message(
                thread_id,
                IncomingMessage(
                    message_id="msg_2",
                    text="try again",
                    already_persisted=True,
                ),
            )
            for _ in range(100):
                st = await orchestrator.get_state(thread_id)
                if st.phase == ResearchPhase.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.02)

            assert st.phase == ResearchPhase.AWAITING_APPROVAL
            assert st.pending_approval is not None
            assert any(
                "temporal" in q.lower() for q in st.pending_approval.sub_questions
            )

