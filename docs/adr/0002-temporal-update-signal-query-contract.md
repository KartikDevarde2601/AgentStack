---
status: accepted
---

# Temporal Update, Signal, and Query Contract for ChatKit User Input

AgentStack bridges interactive ChatKit user actions (`respond`, dynamic clarifying forms, plan approval cards, resume, and thread deletion) to the per-Thread Entity Workflow (`AgentThreadWorkflow`). Inspired by Temporal's interactive research pattern (`InteractiveResearchWorkflow`), we assign each interaction to a specific Temporal primitive based on whether it requires synchronous validation, workflow revival, fire-and-forget control, or read-only status inspection.

## Decision Details

- **Validated HITL Actions → `@workflow.update` + `@*.validator`**:
  - `submit_clarifications(submission: ClarificationSubmission) -> ThreadWorkflowState` (triggered by `research.submit_clarifications` / `research.skip_clarifications`).
  - `submit_approval(decision: ApprovalDecision) -> ThreadWorkflowState` (triggered by `research.approve_plan` / `research.reject_plan`).
  - **Validators** enforce phase (`AWAITING_CLARIFICATIONS` / `AWAITING_APPROVAL`), `turn_id == active_turn_id` (rejecting stale cards from earlier Turns), and single-submission idempotency (rejecting double-clicks) *before* any event is written to workflow history.
  - **Fast, pure handlers**: Update handlers only validate, mutate workflow state to unblock `workflow.wait_condition(...)`, and return `self.get_state()` immediately. All LLM/agent work stays inside `@workflow.run` and streams live deltas through `ThreadEventBus` so ChatKit actions never block or time out on LLM latency.
  - **Graceful rejection**: `TemporalThreadOrchestrator` translates `WorkflowUpdateFailedError` into `WorkflowActionRejectedError`, which `DeepResearchChatKitServer` turns into an immediate assistant notice rather than hanging the SSE stream.
- **Turn Start & Lifecycle Control → `@workflow.signal`**:
  - `submit_message(msg: IncomingMessage)` uses `signal_with_start` (`start_workflow(..., start_signal="submit_message")`) so new prompts and follow-up actions transparently revive a Thread workflow after its 30-minute idle exit, while remaining compatible with both production Temporal Server and fast time-skipping test environments.
  - `end_workflow()` is a fire-and-forget lifecycle signal that unblocks idle or paused `wait_condition`s, prunes ephemeral `run_events`, and exits cleanly (e.g., on `DELETE /api/threads/{thread_id}`).
- **Status & Resume Inspection → `@workflow.query`**:
  - `get_state() -> ThreadWorkflowState` is the sole read-only snapshot used by `GET /api/threads/{thread_id}/state`, `ChatKitServer.respond()` pre-flight guards, and `research.resume` tab-reconnection without adding workflow history events.

## Considered Options

- **Signals for all inputs**: Leaves stale-card clicks and double-submits unvalidated, writing stray events to history and risking indefinite SSE hangs in `ChatKitServer.action()`.
- **`execute_update_with_start_workflow` for `submit_message`**: Adds server-side validation on Turn start in a single RPC, but requires a full local Temporal server instead of `WorkflowEnvironment.start_time_skipping()` for unit tests. Can be adopted later without changing the ChatKit or domain contract.
- **Running Triage/LLM calls inside `@workflow.update` (Demo 4 CLI style)**: Fits synchronous CLI scripts, but delays ChatKit SSE streaming and risks Temporal Update timeouts during slow model calls.

## Consequences

- ChatKit custom action payloads include `workflow_id` (the Turn ID) so validators can reject actions from superseded Turns.
- `ThreadOrchestrator.submit_clarifications` and `submit_approval` return `ThreadWorkflowState` and raise `WorkflowActionRejectedError` on validator rejection.
- Deleting a Thread signals `end_workflow()` on a best-effort basis so parked HITL workflows do not linger until idle timeout.
