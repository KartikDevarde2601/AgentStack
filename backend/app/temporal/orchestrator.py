from __future__ import annotations

from typing import Protocol

from temporalio.client import Client
from temporalio.service import RPCError, RPCStatusCode

from app.domain import (
    ApprovalDecision,
    IncomingMessage,
    ThreadWorkflowState,
)
from app.temporal.workflows import AgentThreadInput, AgentThreadWorkflow


def thread_workflow_id(thread_id: str) -> str:
    return f"thread-{thread_id}"


class ThreadOrchestrator(Protocol):
    async def send_message(
        self, thread_id: str, msg: IncomingMessage
    ) -> ThreadWorkflowState: ...

    async def submit_approval(
        self, thread_id: str, decision: ApprovalDecision
    ) -> None: ...

    async def get_state(self, thread_id: str) -> ThreadWorkflowState: ...


class TemporalThreadOrchestrator:
    """Deep adapter wrapping Temporal signal_with_start, signal, and query."""

    def __init__(
        self,
        client: Client,
        task_queue: str,
        *,
        require_approval: bool = True,
    ) -> None:
        self._client = client
        self._task_queue = task_queue
        self._require_approval = require_approval

    async def send_message(
        self, thread_id: str, msg: IncomingMessage
    ) -> ThreadWorkflowState:
        wf_id = thread_workflow_id(thread_id)
        handle = await self._client.start_workflow(
            AgentThreadWorkflow.run,
            AgentThreadInput(
                thread_id=thread_id,
                require_approval=self._require_approval,
            ),
            id=wf_id,
            task_queue=self._task_queue,
            start_signal="submit_message",
            start_signal_args=[msg],
        )
        return await handle.query(AgentThreadWorkflow.get_state)

    async def submit_approval(
        self, thread_id: str, decision: ApprovalDecision
    ) -> None:
        handle = self._client.get_workflow_handle(thread_workflow_id(thread_id))
        await handle.signal(AgentThreadWorkflow.submit_approval, decision)

    async def get_state(self, thread_id: str) -> ThreadWorkflowState:
        handle = self._client.get_workflow_handle(thread_workflow_id(thread_id))
        try:
            return await handle.query(AgentThreadWorkflow.get_state)
        except RPCError as exc:
            if exc.status in (RPCStatusCode.NOT_FOUND, RPCStatusCode.FAILED_PRECONDITION):
                return ThreadWorkflowState(thread_id=thread_id)
            raise
