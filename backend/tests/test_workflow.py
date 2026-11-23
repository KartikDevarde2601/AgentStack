from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from chatkit.types import AssistantMessageItem, ThreadItem
from temporalio.common import RetryPolicy
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin
from temporalio.contrib.openai_agents.testing import TestModelProvider
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.domain import (
    ApprovalDecision,
    ClarificationSubmission,
    IncomingMessage,
    RequestContext,
    ResearchPhase,
    ResearchRunRecord,
)
from app.events import InMemoryThreadEventBus, ThreadEventKind
from app.temporal.activities import ThreadActivities
from app.temporal.orchestrator import (
    TemporalThreadOrchestrator,
    WorkflowActionRejectedError,
    thread_workflow_id,
)
from app.temporal.workflows import AgentThreadInput, AgentThreadWorkflow
from tests.fake_agents_model import RoutingFakeModel


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


class FakeConversationClient:
    def __init__(
        self,
        *,
        conversation_history: dict[str, list[str]] | None = None,
        fail_first_n_flushes: int = 0,
    ) -> None:
        self.created_conversations: list[str] = []
        self.conversation_history = (
            conversation_history if conversation_history is not None else {}
        )
        self._remaining_flush_failures = fail_first_n_flushes

    async def create_conversation(self) -> str:
        conv_id = f"conv_{uuid.uuid4().hex[:8]}"
        self.created_conversations.append(conv_id)
        self.conversation_history.setdefault(conv_id, [])
        return conv_id

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


def _all_activities(activities: ThreadActivities) -> list:
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
async def test_direct_answer_skips_research_and_keeps_same_thread() -> None:
    store = FakeThreadStore()
    bus = InMemoryThreadEventBus()
    conv_client = FakeConversationClient()
    fake_model = RoutingFakeModel(
        direct_answers={"What does WAL stand for?": "Write-ahead logging."}
    )
    activities = ThreadActivities(client=conv_client, store=store, event_bus=bus)
    thread_id = "thr_direct_answer"

    plugin = OpenAIAgentsPlugin(model_provider=TestModelProvider(fake_model))
    async with await WorkflowEnvironment.start_time_skipping(plugins=[plugin]) as env:
        task_queue = f"test-direct-{uuid.uuid4().hex[:8]}"
        orchestrator = TemporalThreadOrchestrator(
            env.client, task_queue=task_queue, require_approval=True
        )
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=_all_activities(activities),
        ):
            after_seq = await bus.latest_seq(thread_id)
            await orchestrator.send_message(
                thread_id,
                IncomingMessage(message_id="msg_1", text="What does WAL stand for?"),
            )
            kinds = []
            async with asyncio.timeout(5):
                async for event in bus.subscribe(thread_id, after_seq=after_seq):
                    kinds.append(event.kind)
                    if event.kind == ThreadEventKind.TURN_END:
                        break

            assert ThreadEventKind.APPROVAL_PAUSE not in kinds
            assert ThreadEventKind.PROGRESS not in kinds
            reply = store.items[thread_id][-1]
            assert isinstance(reply, AssistantMessageItem)
            assert reply.content[0].text == "Write-ahead logging."
            assert not store.runs

            async with asyncio.timeout(5):
                state = await orchestrator.get_state(thread_id)
                while state.phase != ResearchPhase.IDLE:
                    await asyncio.sleep(0.02)
                    state = await orchestrator.get_state(thread_id)

            # Follow-up specific research query enters the research pipeline on the same thread.
            await orchestrator.send_message(
                thread_id,
                IncomingMessage(
                    message_id="msg_2", text="Compare Postgres and SQLite WAL internals"
                ),
            )
            async with asyncio.timeout(5):
                state = await orchestrator.get_state(thread_id)
                while state.phase != ResearchPhase.AWAITING_APPROVAL:
                    await asyncio.sleep(0.02)
                    state = await orchestrator.get_state(thread_id)

            assert state.pending_approval is not None
            assert not fake_model.searched_queries


@pytest.mark.asyncio
async def test_two_stage_hitl_clarifications_then_plan_approval_and_web_search() -> None:
    store = FakeThreadStore()
    bus = InMemoryThreadEventBus()
    conv_client = FakeConversationClient()
    fake_model = RoutingFakeModel(clarify_keywords=("best",))
    activities = ThreadActivities(client=conv_client, store=store, event_bus=bus)

    plugin = OpenAIAgentsPlugin(model_provider=TestModelProvider(fake_model))
    async with await WorkflowEnvironment.start_time_skipping(plugins=[plugin]) as env:
        task_queue = f"test-two-stage-{uuid.uuid4().hex[:8]}"
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
                    text="Best databases for AI agents",
                    already_persisted=True,
                ),
            )

            # Stage 1: Wait for dynamic Clarifying Questions Gate
            async with asyncio.timeout(5):
                state = await orchestrator.get_state(thread_id)
                while state.phase != ResearchPhase.AWAITING_CLARIFICATIONS:
                    await asyncio.sleep(0.02)
                    state = await orchestrator.get_state(thread_id)

            assert state.pending_clarification is not None
            assert len(state.pending_clarification.questions) == 3
            assert [q.kind for q in state.pending_clarification.questions] == [
                "single_select",
                "multi_select",
                "long_text",
            ]

            # Submit dynamic clarification answers
            await orchestrator.submit_clarifications(
                thread_id,
                ClarificationSubmission(
                    skipped=False,
                    answers={
                        "q_0": "production",
                        "q_1": "Latency, Operations",
                        "q_2": "High-concurrency multi-tenant SaaS",
                    },
                ),
            )

            # Stage 2: InstructionAgent enhances prompt -> PlannerAgent -> Plan Approval Gate
            async with asyncio.timeout(5):
                state = await orchestrator.get_state(thread_id)
                while state.phase != ResearchPhase.AWAITING_APPROVAL:
                    await asyncio.sleep(0.02)
                    state = await orchestrator.get_state(thread_id)

            assert state.pending_approval is not None
            assert len(fake_model.enhanced_prompts) == 1
            assert "High-concurrency multi-tenant SaaS" in fake_model.enhanced_prompts[0]

            # Approve plan with edited queries + extra focus + deep depth
            await orchestrator.submit_approval(
                thread_id,
                ApprovalDecision(
                    approved=True,
                    sub_questions=state.pending_approval.sub_questions[:1],
                    depth="deep",
                    extra_focus="Postgres LISTEN/NOTIFY scaling",
                ),
            )

            # Wait for parallel web searches & report synthesis to finish
            async with asyncio.timeout(5):
                state = await orchestrator.get_state(thread_id)
                while state.phase != ResearchPhase.IDLE:
                    await asyncio.sleep(0.02)
                    state = await orchestrator.get_state(thread_id)

            assert state.turns_since_continue == 1
            assert "Postgres LISTEN/NOTIFY scaling" in fake_model.searched_queries

            events = list(bus._events_by_thread[thread_id])
            assert any(ev.kind == ThreadEventKind.APPROVAL_PAUSE for ev in events)
            assert any(ev.kind == ThreadEventKind.TEXT_DELTA for ev in events)
            assert any(ev.kind == ThreadEventKind.TURN_END for ev in events)

            assistant_items = [
                i for i in store.items[thread_id] if isinstance(i, AssistantMessageItem)
            ]
            assert len(assistant_items) == 1
            assert "# Deep Research Report:" in assistant_items[0].content[0].text


@pytest.mark.asyncio
async def test_reject_plan_and_idle_prune() -> None:
    store = FakeThreadStore()
    bus = InMemoryThreadEventBus()
    conv_client = FakeConversationClient()
    fake_model = RoutingFakeModel()
    activities = ThreadActivities(client=conv_client, store=store, event_bus=bus)

    plugin = OpenAIAgentsPlugin(model_provider=TestModelProvider(fake_model))
    async with await WorkflowEnvironment.start_time_skipping(plugins=[plugin]) as env:
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
                        text="Compare Temporal and Celery",
                        already_persisted=True,
                    )
                ],
            )

            async with asyncio.timeout(5):
                state = await handle.query(AgentThreadWorkflow.get_state)
                while state.phase != ResearchPhase.AWAITING_APPROVAL:
                    await asyncio.sleep(0.02)
                    state = await handle.query(AgentThreadWorkflow.get_state)

            await handle.execute_update(
                AgentThreadWorkflow.submit_approval,
                ApprovalDecision(approved=False, reason="Cancelled by user"),
            )

            await handle.result()

            cancelled_items = [
                i
                for i in store.items[thread_id]
                if isinstance(i, AssistantMessageItem)
                and "Research Cancelled" in i.content[0].text
            ]
            assert len(cancelled_items) == 1
            assert await bus.latest_seq(thread_id) == 0


@pytest.mark.asyncio
async def test_retry_after_failed_turn_preserves_original_topic() -> None:
    shared_history: dict[str, list[str]] = {}
    store = FakeThreadStore()
    bus = InMemoryThreadEventBus()
    conv_client = FakeConversationClient(
        conversation_history=shared_history, fail_first_n_flushes=1
    )
    fake_model = RoutingFakeModel(
        fail_first_n_triage=1,
        conversation_history=shared_history,
    )
    activities = ThreadActivities(client=conv_client, store=store, event_bus=bus)

    plugin = OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            retry_policy=RetryPolicy(maximum_attempts=1)
        ),
        model_provider=TestModelProvider(fake_model),
    )
    async with await WorkflowEnvironment.start_time_skipping(plugins=[plugin]) as env:
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

            await orchestrator.send_message(
                thread_id,
                IncomingMessage(
                    message_id="msg_1",
                    text="Explain how Temporal durable execution improves reliability.",
                    already_persisted=True,
                ),
            )
            async with asyncio.timeout(5):
                st = await orchestrator.get_state(thread_id)
                while not (st.phase == ResearchPhase.IDLE and st.turns_since_continue == 1):
                    await asyncio.sleep(0.02)
                    st = await orchestrator.get_state(thread_id)

            await orchestrator.send_message(
                thread_id,
                IncomingMessage(
                    message_id="msg_2",
                    text="try again",
                    already_persisted=True,
                ),
            )
            async with asyncio.timeout(5):
                st = await orchestrator.get_state(thread_id)
                while st.phase != ResearchPhase.AWAITING_APPROVAL:
                    await asyncio.sleep(0.02)
                    st = await orchestrator.get_state(thread_id)

            assert st.pending_approval is not None
            assert any(
                "temporal" in q.lower() for q in st.pending_approval.sub_questions
            )


@pytest.mark.asyncio
async def test_update_validators_reject_stale_actions_and_end_workflow_exits_cleanly() -> None:
    store = FakeThreadStore()
    bus = InMemoryThreadEventBus()
    conv_client = FakeConversationClient()
    fake_model = RoutingFakeModel(clarify_keywords=("best",))
    activities = ThreadActivities(client=conv_client, store=store, event_bus=bus)

    plugin = OpenAIAgentsPlugin(model_provider=TestModelProvider(fake_model))
    async with await WorkflowEnvironment.start_time_skipping(plugins=[plugin]) as env:
        task_queue = f"test-validators-{uuid.uuid4().hex[:8]}"
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
                    text="Best vector databases",
                    already_persisted=True,
                ),
            )
            async with asyncio.timeout(5):
                state = await orchestrator.get_state(thread_id)
                while state.phase != ResearchPhase.AWAITING_CLARIFICATIONS:
                    await asyncio.sleep(0.02)
                    state = await orchestrator.get_state(thread_id)

            # Wrong phase: submit_approval while awaiting clarifications must be rejected.
            with pytest.raises(WorkflowActionRejectedError, match="not currently awaiting plan approval"):
                await orchestrator.submit_approval(
                    thread_id, ApprovalDecision(approved=True)
                )

            # Stale turn_id on clarifications must be rejected.
            with pytest.raises(WorkflowActionRejectedError, match="earlier turn"):
                await orchestrator.submit_clarifications(
                    thread_id,
                    ClarificationSubmission(
                        turn_id="turn_stale_999",
                        skipped=False,
                        answers={"q_0": "production"},
                    ),
                )

            # Empty non-skipped answers must be rejected.
            with pytest.raises(WorkflowActionRejectedError, match="at least one clarification"):
                await orchestrator.submit_clarifications(
                    thread_id,
                    ClarificationSubmission(
                        turn_id=state.active_turn_id,
                        skipped=False,
                        answers={},
                    ),
                )

            # Fire-and-forget end_workflow signal cleanly exits parked workflow & prunes events.
            await orchestrator.end_workflow(thread_id)
            handle = env.client.get_workflow_handle(thread_workflow_id(thread_id))
            await handle.result()
            assert await bus.latest_seq(thread_id) == 0
