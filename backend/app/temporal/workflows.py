from __future__ import annotations

import asyncio
from datetime import timedelta

from pydantic import BaseModel, Field
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from app.domain import (
        ApprovalDecision,
        ConversationReply,
        IncomingMessage,
        PendingApproval,
        ResearchFinding,
        ResearchPhase,
        ResearchPlan,
        ThreadWorkflowState,
        TurnResponse,
    )
    from app.temporal.activities import (
        GatherFindingInput,
        PersistApprovalWidgetInput,
        PersistFinalItemInput,
        PersistUserMessageInput,
        PlanResearchInput,
        PublishProgressInput,
        PublishTurnEndInput,
        RecordFailedTurnsInput,
        SynthesizeReportInput,
    )

_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_attempts=3,
)

DEFAULT_IDLE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes
DEFAULT_MAX_TURNS_BEFORE_CONTINUE = 25


class AgentThreadInput(BaseModel):
    thread_id: str
    openai_conversation_id: str | None = None
    unrecorded_turns: list[tuple[str, str]] = Field(default_factory=list)
    require_approval: bool = True
    idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS
    max_turns_before_continue: int = DEFAULT_MAX_TURNS_BEFORE_CONTINUE


@workflow.defn(name="AgentThreadWorkflow")
class AgentThreadWorkflow:
    """Durable per-Thread Entity Workflow (id = thread-{thread_id}).

    Owns turn execution, Approval Gate suspension, 30-min idle exit,
    and continue_as_new history bounds.
    """

    def __init__(self) -> None:
        self._thread_id: str = ""
        self._openai_conversation_id: str | None = None
        self._phase: ResearchPhase = ResearchPhase.IDLE
        self._active_turn_id: str | None = None
        self._message: str = ""
        self._pending_message: IncomingMessage | None = None
        self._pending_approval: PendingApproval | None = None
        self._approval_decision: ApprovalDecision | None = None
        self._turns_since_continue: int = 0
        self._active_query: str = ""
        self._active_sub_questions: list[str] = []
        self._unrecorded_turns: list[tuple[str, str]] = []

    @workflow.signal
    def submit_message(self, msg: IncomingMessage) -> None:
        if self._active_turn_id is None and self._pending_message is None:
            self._pending_message = msg

    @workflow.signal
    def submit_approval(self, decision: ApprovalDecision) -> None:
        self._approval_decision = decision

    @workflow.query
    def get_state(self) -> ThreadWorkflowState:
        return ThreadWorkflowState(
            thread_id=self._thread_id,
            openai_conversation_id=self._openai_conversation_id,
            phase=self._phase,
            active_turn_id=self._active_turn_id,
            message=self._message,
            pending_approval=self._pending_approval,
            turns_since_continue=self._turns_since_continue,
        )

    @workflow.run
    async def run(self, params: AgentThreadInput) -> None:
        self._thread_id = params.thread_id
        self._openai_conversation_id = params.openai_conversation_id
        self._unrecorded_turns.extend(params.unrecorded_turns)

        while True:
            if self._pending_message is None:
                self._phase = ResearchPhase.IDLE
                self._active_turn_id = None
                self._message = ""
                try:
                    await workflow.wait_condition(
                        lambda: self._pending_message is not None,
                        timeout=timedelta(seconds=params.idle_timeout_seconds),
                    )
                except TimeoutError:
                    await workflow.execute_activity(
                        "prune_thread_events_activity",
                        self._thread_id,
                        result_type=int,
                        start_to_close_timeout=timedelta(seconds=30),
                        retry_policy=_ACTIVITY_RETRY,
                    )
                    return

            if self._openai_conversation_id is None:
                self._openai_conversation_id = await workflow.execute_activity(
                    "ensure_openai_conversation_activity",
                    self._thread_id,
                    result_type=str,
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )

            next_msg = self._pending_message
            assert next_msg is not None
            self._pending_message = None
            turn_id = f"turn_{workflow.uuid4().hex[:10]}"
            self._approval_decision = None
            self._pending_approval = None
            self._active_turn_id = turn_id
            self._active_query = next_msg.text
            self._active_sub_questions = []

            try:
                await self._run_turn(
                    msg=next_msg,
                    turn_id=turn_id,
                    require_approval=params.require_approval,
                )
            finally:
                self._pending_approval = None

            self._turns_since_continue += 1
            if (
                workflow.info().is_continue_as_new_suggested()
                or self._turns_since_continue >= params.max_turns_before_continue
            ):
                workflow.continue_as_new(
                    AgentThreadInput(
                        thread_id=self._thread_id,
                        openai_conversation_id=self._openai_conversation_id,
                        unrecorded_turns=list(self._unrecorded_turns),
                        require_approval=params.require_approval,
                        idle_timeout_seconds=params.idle_timeout_seconds,
                        max_turns_before_continue=params.max_turns_before_continue,
                    )
                )

    async def _flush_unrecorded_turns(self, *, best_effort: bool = False) -> None:
        if not self._unrecorded_turns or not self._openai_conversation_id:
            return
        retry_policy = (
            RetryPolicy(maximum_attempts=1) if best_effort else _ACTIVITY_RETRY
        )
        await workflow.execute_activity(
            "record_failed_turns_activity",
            RecordFailedTurnsInput(
                conversation_id=self._openai_conversation_id,
                turns=list(self._unrecorded_turns),
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=retry_policy,
        )
        self._unrecorded_turns.clear()

    async def _run_turn(
        self,
        *,
        msg: IncomingMessage,
        turn_id: str,
        require_approval: bool,
    ) -> None:
        query = msg.text
        # Temporal caches patch decisions per workflow run. Scope the marker to
        # this turn so replaying an older turn does not pin subsequent turns.
        resolve_turn = workflow.patched(
            f"resolve-conversation-turn-{self._turns_since_continue}"
        )
        response_received = False
        research_started = not resolve_turn
        try:
            if not msg.already_persisted:
                await workflow.execute_activity(
                    "persist_user_message_activity",
                    PersistUserMessageInput(
                        thread_id=self._thread_id,
                        turn_id=turn_id,
                        message_id=msg.message_id,
                        text=query,
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )

            plan_input = PlanResearchInput(
                query=query, conversation_id=self._openai_conversation_id
            )
            if resolve_turn:
                self._phase = ResearchPhase.PLANNING
                self._message = "Reading your request..."
                await self._flush_unrecorded_turns()
                response = await workflow.execute_activity(
                    "resolve_turn_activity",
                    plan_input,
                    result_type=TurnResponse,
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=_ACTIVITY_RETRY,
                )
                response_received = True
                if isinstance(response, ConversationReply):
                    await workflow.execute_activity(
                        "persist_final_item_activity",
                        PersistFinalItemInput(
                            thread_id=self._thread_id,
                            turn_id=turn_id,
                            query=query,
                            phase=ResearchPhase.COMPLETED,
                            message="",
                            text=response.text,
                            record_research_run=False,
                        ),
                        start_to_close_timeout=timedelta(seconds=30),
                        retry_policy=_ACTIVITY_RETRY,
                    )
                    self._phase = ResearchPhase.COMPLETED
                    self._message = ""
                    return
                plan = response
                research_started = True

            self._phase = ResearchPhase.PLANNING
            self._message = "Planning research sub-questions..."
            await workflow.execute_activity(
                "publish_progress_activity",
                PublishProgressInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=query,
                    phase=ResearchPhase.PLANNING,
                    message=self._message,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )

            if not resolve_turn:
                # Replay the planner for turns already started by an older worker.
                await self._flush_unrecorded_turns()
                plan = await workflow.execute_activity(
                    "plan_research_activity",
                    plan_input,
                    result_type=ResearchPlan,
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=_ACTIVITY_RETRY,
                )
                response_received = True
            query = plan.query or query
            self._active_query = query
            sub_questions = list(plan.sub_questions)
            self._active_sub_questions = list(sub_questions)
            selected_depth = "standard"

            if require_approval:
                approval_id = f"apr_{workflow.uuid4().hex[:12]}"
                widget_id = f"wgt_approval_{turn_id}"
                self._pending_approval = PendingApproval(
                    turn_id=turn_id,
                    approval_id=approval_id,
                    sub_questions=sub_questions,
                    widget_item_id=widget_id,
                )
                self._phase = ResearchPhase.AWAITING_APPROVAL
                self._message = "Waiting for your approval on the research plan..."

                await workflow.execute_activity(
                    "publish_progress_activity",
                    PublishProgressInput(
                        thread_id=self._thread_id,
                        turn_id=turn_id,
                        query=query,
                        phase=ResearchPhase.AWAITING_APPROVAL,
                        message=self._message,
                        sub_questions=sub_questions,
                        total_sub_questions=len(sub_questions),
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )

                await workflow.execute_activity(
                    "persist_approval_widget_activity",
                    PersistApprovalWidgetInput(
                        thread_id=self._thread_id,
                        turn_id=turn_id,
                        sub_questions=sub_questions,
                        state="pending",
                    ),
                    result_type=str,
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )

                await workflow.wait_condition(
                    lambda: self._approval_decision is not None
                )

                decision = self._approval_decision
                assert decision is not None
                self._pending_approval = None

                if not decision.approved:
                    await workflow.execute_activity(
                        "persist_approval_widget_activity",
                        PersistApprovalWidgetInput(
                            thread_id=self._thread_id,
                            turn_id=turn_id,
                            sub_questions=sub_questions,
                            state="cancelled",
                        ),
                        result_type=str,
                        start_to_close_timeout=timedelta(seconds=30),
                        retry_policy=_ACTIVITY_RETRY,
                    )
                    reason = decision.reason or "Research plan cancelled by user."
                    self._phase = ResearchPhase.COMPLETED
                    self._message = "Research plan cancelled by user."
                    await workflow.execute_activity(
                        "persist_final_item_activity",
                        PersistFinalItemInput(
                            thread_id=self._thread_id,
                            turn_id=turn_id,
                            query=query,
                            phase=ResearchPhase.COMPLETED,
                            message=self._message,
                            sub_questions=sub_questions,
                            text=(
                                "### Research Cancelled\n\n"
                                f"Research for **{query}** was cancelled at the approval step.\n\n"
                                f"_{reason}_"
                            ),
                        ),
                        start_to_close_timeout=timedelta(seconds=30),
                        retry_policy=_ACTIVITY_RETRY,
                    )
                    return

                if decision.sub_questions:
                    sub_questions = list(decision.sub_questions)
                extra = decision.extra_focus.strip()
                if extra and extra not in sub_questions:
                    sub_questions.append(extra)
                selected_depth = decision.depth
                self._active_sub_questions = list(sub_questions)

                await workflow.execute_activity(
                    "persist_approval_widget_activity",
                    PersistApprovalWidgetInput(
                        thread_id=self._thread_id,
                        turn_id=turn_id,
                        sub_questions=sub_questions,
                        state="approved",
                        selected_depth=decision.depth,
                        extra_focus=decision.extra_focus,
                    ),
                    result_type=str,
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )

            total = len(sub_questions)
            self._phase = ResearchPhase.RESEARCHING
            self._message = f"Researching {total} sub-questions in parallel (0/{total})..."
            await workflow.execute_activity(
                "publish_progress_activity",
                PublishProgressInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=query,
                    phase=ResearchPhase.RESEARCHING,
                    message=self._message,
                    sub_questions=sub_questions,
                    completed_sub_questions=0,
                    total_sub_questions=total,
                    update_progress_widget=True,
                    is_new_progress_widget=True,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )

            completed_count = 0

            async def _investigate(sub_q: str) -> ResearchFinding:
                nonlocal completed_count
                finding: ResearchFinding = await workflow.execute_activity(
                    "investigate_sub_question_activity",
                    GatherFindingInput(
                        query=query,
                        sub_question=sub_q,
                        depth=selected_depth,
                        conversation_id=self._openai_conversation_id,
                    ),
                    result_type=ResearchFinding,
                    start_to_close_timeout=timedelta(seconds=90),
                    retry_policy=_ACTIVITY_RETRY,
                )

                completed_count += 1
                self._message = (
                    f"Researching sub-questions ({completed_count}/{total} completed)..."
                )
                await workflow.execute_activity(
                    "publish_progress_activity",
                    PublishProgressInput(
                        thread_id=self._thread_id,
                        turn_id=turn_id,
                        query=query,
                        phase=ResearchPhase.RESEARCHING,
                        message=self._message,
                        sub_questions=sub_questions,
                        completed_sub_questions=completed_count,
                        total_sub_questions=total,
                        update_progress_widget=True,
                        is_new_progress_widget=False,
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )
                return finding

            findings = list(
                await asyncio.gather(*(_investigate(q) for q in sub_questions))
            )

            self._phase = ResearchPhase.SYNTHESIZING
            self._message = "Synthesizing findings into a structured report..."
            await workflow.execute_activity(
                "publish_progress_activity",
                PublishProgressInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=query,
                    phase=ResearchPhase.SYNTHESIZING,
                    message=self._message,
                    sub_questions=sub_questions,
                    completed_sub_questions=total,
                    total_sub_questions=total,
                    update_progress_widget=True,
                    is_new_progress_widget=False,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )

            await workflow.execute_activity(
                "synthesize_report_activity",
                SynthesizeReportInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=query,
                    findings=findings,
                    conversation_id=self._openai_conversation_id,
                ),
                result_type=str,
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=_ACTIVITY_RETRY,
            )

            self._phase = ResearchPhase.COMPLETED
            self._message = "Deep research completed."
            await workflow.execute_activity(
                "publish_progress_activity",
                PublishProgressInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=query,
                    phase=ResearchPhase.COMPLETED,
                    message=self._message,
                    sub_questions=sub_questions,
                    completed_sub_questions=total,
                    total_sub_questions=total,
                    update_progress_widget=True,
                    is_new_progress_widget=False,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )
            await workflow.execute_activity(
                "publish_turn_end_activity",
                PublishTurnEndInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    phase=ResearchPhase.COMPLETED,
                    message=self._message,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )
        except Exception as exc:
            if not response_received:
                self._unrecorded_turns.append((query, str(exc)))
                try:
                    await self._flush_unrecorded_turns(best_effort=True)
                except Exception:
                    # Keep in durable workflow state to flush on the next Turn.
                    pass
            self._phase = ResearchPhase.FAILED
            if research_started:
                self._message = f"Deep research failed: {exc}"
                failure_text = (
                    "### Research Failed\n\n"
                    f"Unable to complete research for **{query}**.\n\n"
                    f"Error: `{exc}`"
                )
            else:
                self._message = "Could not respond to your request."
                failure_text = "I couldn't respond to your request. Please try again."
            await workflow.execute_activity(
                "persist_final_item_activity",
                PersistFinalItemInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=query,
                    phase=ResearchPhase.FAILED,
                    message=self._message,
                    sub_questions=self._active_sub_questions,
                    text=failure_text,
                    error=str(exc),
                    record_research_run=research_started,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )
