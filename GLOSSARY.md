# AgentStack

AgentStack is a conversational deep-research workspace where users collaborate with an autonomous research agent across multi-turn conversations and review and adjust research plans before execution.

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
