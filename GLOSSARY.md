# AgentStack

AgentStack is a conversational deep-research workspace where users collaborate with an autonomous research agent across multi-turn conversations, review and adjust research plans before execution, and intervene while work is in progress.

## Language

**Thread**:
A persistent research conversation with a user, containing an ordered timeline of Turns and interactive cards.
_Avoid_: Session, chat, room

**Turn**:
A single user request and the agent's complete response cycle, including any planning, approval, parallel investigation, and synthesis.
_Avoid_: Run, execution, request

**Research Plan**:
The proposed set of sub-questions and depth settings produced for a Turn before parallel investigation begins.
_Avoid_: Strategy, outline, breakdown

**Approval Gate**:
A pause point during a Turn where the agent waits for explicit human confirmation, modification, or cancellation of a Research Plan before proceeding.
_Avoid_: HITL check, confirmation modal, prompt gate

**Finding**:
Structured evidence gathered for one sub-question of a Research Plan, consisting of a summary, key points, and cited sources.
_Avoid_: Result, snippet, search output

**Queued Message**:
A follow-up user message submitted while a Turn is active, held in order until the active Turn finishes and then started as the next Turn.
_Avoid_: Pending prompt, deferred message

**Steer**:
A mid-Turn user intervention that cancels in-flight work, preserves any partial output as interrupted, and immediately redirects the agent on a new Turn.
_Avoid_: Override, abort-and-resend, hot-inject
