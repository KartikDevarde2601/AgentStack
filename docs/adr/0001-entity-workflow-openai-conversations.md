---
status: accepted
---

# Per-Thread Entity Workflow with OpenAI-Managed Conversation History and Postgres Stream Bridge

AgentStack integrates three systems—self-hosted ChatKit, Temporal, and the OpenAI Agents SDK—for durable, multi-turn deep research with Approval Gates and mid-Turn user intervention (Queued Messages and Steers). We decided to run **one durable Temporal Entity Workflow per Thread** (`AgentThreadWorkflow`, `id = thread-{thread_id}`), delegate **LLM conversation memory exclusively to the OpenAI Conversations API** (`openai_conversation_id` stored 1:1 on `chatkit_threads`), and stream live deltas from Temporal activities to FastAPI via a deep **`ThreadEventBus` module** backed by Postgres `run_events` + `LISTEN/NOTIFY`.

## Decision Details

- **Single write authority per concern**:
  - **Temporal (`AgentThreadWorkflow`)** is the sole orchestrator and turn serializer per Thread (owns active Turn state, open tool `call_id`s, Approval Gate `wait_condition`, and FIFO inbox).
  - **OpenAI Conversations API** owns LLM memory and is mutated exclusively inside Temporal activities via `OpenAIAgentsPlugin` and `activity_as_tool`. FastAPI never writes to OpenAI directly.
  - **Postgres** owns the permanent ChatKit UI read-model (`chatkit_threads`, `chatkit_items`), `thread_id ↔ openai_conversation_id` mapping, and the ephemeral `run_events` stream buffer.
- **Durable tool-call Approval Gate**: `request_plan_approval` is a workflow-local `@function_tool` (not an activity) that emits the approval widget, suspends on `workflow.wait_condition(...)` with zero compute held, and returns `ApprovalDecision` as the native `function_call_output` when signaled by `ChatKitServer.action()`.
- **Strict `call_id` closure**: On a **Steer** or a new message while paused at an Approval Gate, `AgentThreadWorkflow` cancels in-flight activities, marks the stale approval widget `Cancelled`, and emits synthetic `{"status": "cancelled_by_user"}` tool outputs for every open `call_id` before starting the next Turn—preventing OpenAI `400` dangling tool-call errors. Partial streamed output from the interrupted Turn is preserved and marked as interrupted.
- **Lifecycle & history bounds**: `AgentThreadWorkflow` exits cleanly after **30 minutes idle** (no active Turn, no pending Approval Gate, empty queue) and revives transparently via `signal_with_start`. During long active sessions, it calls `continue_as_new` at a safe Turn boundary when `is_continue_as_new_suggested()` is true or `turns_since_continue >= 25`, carrying forward only `thread_id`, `openai_conversation_id`, and queued messages.
- **Mid-Turn UI**: Because `@openai/chatkit-react` locks its built-in composer during active SSE streams, `ResearchChatPane` replaces its static background banner with a compact **Mid-Run Control Bar** offering **Queue** (shown as removable pending chips and persisted to `chatkit_items` only when the Turn starts) and **Steer** (interrupts and pivots immediately), posting to `POST /api/threads/{thread_id}/messages`.
- **No runtime offline demo mode**: `OPENAI_API_KEY` is required at runtime (eliminating silent template fallbacks). Automated tests use an in-memory `ThreadEventBus` and mocked/fake adapters strictly inside `pytest`.

## Considered Options

- **Per-Turn Temporal Workflow**: Simpler deploys without `continue_as_new`, but requires external Postgres locking/queuing and risks race conditions that leave dangling `function_call`s in OpenAI Conversations on mid-Turn interrupts.
- **250ms Temporal Query Polling / Redis Streams**: Polling cannot deliver live token deltas and adds load; Redis adds new infrastructure when Postgres 16 is already required.
- **Runtime Offline Demo Fallback**: Previously hid missing API keys behind hardcoded sub-question templates; removed in favor of failing fast when `OPENAI_API_KEY` is absent.

## Consequences

- `run_events` is strictly an ephemeral stream buffer—prune rows on workflow idle-exit or after 24 hours; permanent UI history lives only in `chatkit_items`.
- Activity retries must tag emitted stream chunks with `(activity_id, attempt)` so retried model/tool activities do not duplicate streamed text in ChatKit.
- Future breaking changes to `AgentThreadWorkflow` must use Temporal `workflow.patched` or Worker Versioning for any Threads parked at multi-hour Approval Gates.
