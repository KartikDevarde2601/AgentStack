from __future__ import annotations

from typing import Protocol

from temporalio.client import Client, WorkflowUpdateFailedError
from temporalio.service import RPCError, RPCStatusCode

from app.domain import (
    ApprovalDecision,
    ClarificationSubmission,
    IncomingMessage,
    ThreadWorkflowState,
)
from app.temporal.workflows import AgentThreadInput, AgentThreadWorkflow


def thread_workflow_id(thread_id: str) -> str:
    return f"thread-{thread_id}"


class WorkflowActionRejectedError(Exception):
    """Raised when a Temporal @workflow.update validator rejects a user action."""


class ThreadOrchestrator(Protocol):
    async def send_message(
        self, thread_id: str, msg: IncomingMessage
    ) -> ThreadWorkflowState: ...

    async def submit_clarifications(
        self, thread_id: str, submission: ClarificationSubmission
    ) -> ThreadWorkflowState: ...

    async def submit_approval(
        self, thread_id: str, decision: ApprovalDecision
    ) -> ThreadWorkflowState: ...

    async def end_workflow(self, thread_id: str) -> None: ...

    async def get_state(self, thread_id: str) -> ThreadWorkflowState: ...


def _extract_rejection_reason(exc: WorkflowUpdateFailedError) -> str:
    cause = exc.cause
    if cause is not None and str(cause).strip():
        return str(cause).strip()
    return str(exc).strip() or "This action is no longer valid for the current thread state."


class TemporalThreadOrchestrator:
    """Deep adapter wrapping Temporal signal_with_start, validated updates, signals, and queries."""

    def __init__(
        self,
        client: Client,
        task_queue: str,
        *,
        model: str = "gpt-4.1-mini",
        require_approval: bool = True,
    ) -> None:
        self._client = client
        self._task_queue = task_queue
        self._model = model
        self._require_approval = require_approval

    async def send_message(
        self, thread_id: str, msg: IncomingMessage
    ) -> ThreadWorkflowState:
        wf_id = thread_workflow_id(thread_id)
        handle = await self._client.start_workflow(
            AgentThreadWorkflow.run,
            AgentThreadInput(
                thread_id=thread_id,
                model=self._model,
                require_approval=self._require_approval,
            ),
            id=wf_id,
            task_queue=self._task_queue,
            start_signal="submit_message",
            start_signal_args=[msg],
        )
        return await handle.query(AgentThreadWorkflow.get_state)

    async def submit_clarifications(
        self, thread_id: str, submission: ClarificationSubmission
    ) -> ThreadWorkflowState:
        handle = self._client.get_workflow_handle(thread_workflow_id(thread_id))
        try:
            return await handle.execute_update(
                AgentThreadWorkflow.submit_clarifications, submission
            )
        except WorkflowUpdateFailedError as exc:
            raise WorkflowActionRejectedError(_extract_rejection_reason(exc)) from exc
        except RPCError as exc:
            if exc.status in (RPCStatusCode.NOT_FOUND, RPCStatusCode.FAILED_PRECONDITION):
                raise WorkflowActionRejectedError(
                    "This research session is no longer active. Please send a new message."
                ) from exc
            raise

    async def submit_approval(
        self, thread_id: str, decision: ApprovalDecision
    ) -> ThreadWorkflowState:
        handle = self._client.get_workflow_handle(thread_workflow_id(thread_id))
        try:
            return await handle.execute_update(
                AgentThreadWorkflow.submit_approval, decision
            )
        except WorkflowUpdateFailedError as exc:
            raise WorkflowActionRejectedError(_extract_rejection_reason(exc)) from exc
        except RPCError as exc:
            if exc.status in (RPCStatusCode.NOT_FOUND, RPCStatusCode.FAILED_PRECONDITION):
                raise WorkflowActionRejectedError(
                    "This research session is no longer active. Please send a new message."
                ) from exc
            raise

    async def end_workflow(self, thread_id: str) -> None:
        handle = self._client.get_workflow_handle(thread_workflow_id(thread_id))
        try:
            await handle.signal(AgentThreadWorkflow.end_workflow)
        except RPCError as exc:
            if exc.status in (RPCStatusCode.NOT_FOUND, RPCStatusCode.FAILED_PRECONDITION):
                return
            raise

    async def get_state(self, thread_id: str) -> ThreadWorkflowState:
        handle = self._client.get_workflow_handle(thread_workflow_id(thread_id))
        try:
            return await handle.query(AgentThreadWorkflow.get_state)
        except RPCError as exc:
            if exc.status in (RPCStatusCode.NOT_FOUND, RPCStatusCode.FAILED_PRECONDITION):
                return ThreadWorkflowState(thread_id=thread_id)
            raise
