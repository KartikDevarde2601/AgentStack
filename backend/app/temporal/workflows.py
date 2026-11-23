from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel, Field
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from app.domain import (
        ApprovalDecision,
        ClarificationSubmission,
        IncomingMessage,
        PendingApproval,
        PendingClarification,
        ResearchFinding,
        ResearchPhase,
        ThreadWorkflowState,
    )
    from app.research.agents.models import EnhancedResearchBrief, WebSearchItem
    from app.research.agents.research_manager import DeepResearchManager
    from app.temporal.activities import (
        PersistApprovalWidgetInput,
        PersistClarificationsWidgetInput,
        PersistFinalItemInput,
        PersistUserMessageInput,
        PublishProgressInput,
        PublishReportInput,
        PublishTurnEndInput,
        RecordFailedTurnsInput,
    )

_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_attempts=3,
)

DEFAULT_IDLE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes
DEFAULT_MAX_TURNS_BEFORE_CONTINUE = 25


class AgentThreadInput(BaseModel):
    thread_id: str
    model: str = "gpt-4.1-mini"
    openai_conversation_id: str | None = None
    unrecorded_turns: list[tuple[str, str]] = Field(default_factory=list)
    require_approval: bool = True
    idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS
    max_turns_before_continue: int = DEFAULT_MAX_TURNS_BEFORE_CONTINUE


@workflow.defn(name="AgentThreadWorkflow")
class AgentThreadWorkflow:
    """Durable per-Thread Entity Workflow (id = thread-{thread_id}).

    Orchestrates the multi-agent DeepResearchManager (Triage -> Clarifying ->
    Instruction -> Planner -> Parallel WebSearch -> Writer) via OpenAIAgentsPlugin,
    with two HITL gates (dynamic clarifications + plan approval), 30-min idle exit,
    and continue_as_new history bounds.
    """

    def __init__(self) -> None:
        self._thread_id: str = ""
        self._model: str = "gpt-4.1-mini"
        self._openai_conversation_id: str | None = None
        self._phase: ResearchPhase = ResearchPhase.IDLE
        self._active_turn_id: str | None = None
        self._message: str = ""
        self._pending_message: IncomingMessage | None = None
        self._pending_clarification: PendingClarification | None = None
        self._clarification_submission: ClarificationSubmission | None = None
        self._pending_approval: PendingApproval | None = None
        self._approval_decision: ApprovalDecision | None = None
        self._turns_since_continue: int = 0
        self._active_query: str = ""
        self._active_sub_questions: list[str] = []
        self._unrecorded_turns: list[tuple[str, str]] = []
        self._workflow_ended: bool = False

    @workflow.signal
    def submit_message(self, msg: IncomingMessage) -> None:
        if (
            not self._workflow_ended
            and self._active_turn_id is None
            and self._pending_message is None
        ):
            self._pending_message = msg

    @workflow.signal
    def end_workflow(self) -> None:
        """Fire-and-forget lifecycle signal to gracefully exit the workflow."""
        self._workflow_ended = True

    @workflow.update
    async def submit_clarifications(
        self, submission: ClarificationSubmission
    ) -> ThreadWorkflowState:
        """Validated HITL update to submit or skip clarifying questions."""
        self._clarification_submission = submission
        return self.get_state()

    @submit_clarifications.validator
    def validate_submit_clarifications(
        self, submission: ClarificationSubmission
    ) -> None:
        if self._workflow_ended:
            raise ValueError("This research thread has ended.")
        if (
            self._phase != ResearchPhase.AWAITING_CLARIFICATIONS
            or self._pending_clarification is None
        ):
            raise ValueError("This thread is not currently awaiting clarifications.")
        if self._clarification_submission is not None:
            raise ValueError("Clarifications have already been submitted for this turn.")
        if submission.turn_id and submission.turn_id != self._active_turn_id:
            raise ValueError(
                "This clarification card belongs to an earlier turn and is no longer active."
            )
        if not submission.skipped and not submission.answers:
            raise ValueError("Please provide at least one clarification answer or choose Skip.")

    @workflow.update
    async def submit_approval(
        self, decision: ApprovalDecision
    ) -> ThreadWorkflowState:
        """Validated HITL update to approve or cancel the planned web searches."""
        self._approval_decision = decision
        return self.get_state()

    @submit_approval.validator
    def validate_submit_approval(self, decision: ApprovalDecision) -> None:
        if self._workflow_ended:
            raise ValueError("This research thread has ended.")
        if (
            self._phase != ResearchPhase.AWAITING_APPROVAL
            or self._pending_approval is None
        ):
            raise ValueError("This thread is not currently awaiting plan approval.")
        if self._approval_decision is not None:
            raise ValueError("A decision has already been recorded for this research plan.")
        if decision.turn_id and decision.turn_id != self._active_turn_id:
            raise ValueError(
                "This approval card belongs to an earlier turn and is no longer active."
            )

    @workflow.query
    def get_state(self) -> ThreadWorkflowState:
        return ThreadWorkflowState(
            thread_id=self._thread_id,
            openai_conversation_id=self._openai_conversation_id,
            phase=self._phase,
            active_turn_id=self._active_turn_id,
            message=self._message,
            pending_clarification=self._pending_clarification,
            pending_approval=self._pending_approval,
            turns_since_continue=self._turns_since_continue,
        )

    @workflow.run
    async def run(self, params: AgentThreadInput) -> None:
        self._thread_id = params.thread_id
        self._model = params.model
        self._openai_conversation_id = params.openai_conversation_id
        self._unrecorded_turns.extend(params.unrecorded_turns)

        while True:
            if self._workflow_ended:
                await self._prune_events()
                return

            if self._pending_message is None:
                self._phase = ResearchPhase.IDLE
                self._active_turn_id = None
                self._message = ""
                try:
                    await workflow.wait_condition(
                        lambda: self._workflow_ended or self._pending_message is not None,
                        timeout=timedelta(seconds=params.idle_timeout_seconds),
                    )
                except TimeoutError:
                    await self._prune_events()
                    return

                if self._workflow_ended:
                    await self._prune_events()
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
            self._clarification_submission = None
            self._pending_clarification = None
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
                self._pending_clarification = None
                self._pending_approval = None

            self._turns_since_continue += 1
            if (
                workflow.info().is_continue_as_new_suggested()
                or self._turns_since_continue >= params.max_turns_before_continue
            ):
                workflow.continue_as_new(
                    AgentThreadInput(
                        thread_id=self._thread_id,
                        model=self._model,
                        openai_conversation_id=self._openai_conversation_id,
                        unrecorded_turns=list(self._unrecorded_turns),
                        require_approval=params.require_approval,
                        idle_timeout_seconds=params.idle_timeout_seconds,
                        max_turns_before_continue=params.max_turns_before_continue,
                    )
                )

    async def _prune_events(self) -> None:
        await workflow.execute_activity(
            "prune_thread_events_activity",
            self._thread_id,
            result_type=int,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_ACTIVITY_RETRY,
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
        manager = DeepResearchManager(model=self._model)
        response_received = False
        research_started = False
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

            self._phase = ResearchPhase.PLANNING
            self._message = "Analyzing your request..."
            await self._flush_unrecorded_turns()

            outcome = await manager.triage(
                query, conversation_id=self._openai_conversation_id
            )
            response_received = True

            # Branch 1: Direct conversational answer
            if outcome.kind == "answer":
                await workflow.execute_activity(
                    "persist_final_item_activity",
                    PersistFinalItemInput(
                        thread_id=self._thread_id,
                        turn_id=turn_id,
                        query=query,
                        phase=ResearchPhase.COMPLETED,
                        message="",
                        text=outcome.direct_answer or "",
                        record_research_run=False,
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )
                self._phase = ResearchPhase.COMPLETED
                self._message = ""
                return

            research_started = True
            brief: EnhancedResearchBrief

            # Branch 2: Dynamic HITL Clarifications -> Prompt Enhancement
            if outcome.kind == "clarify" and outcome.clarifications is not None:
                clarifications = outcome.clarifications
                clarification_id = f"clr_{workflow.uuid4().hex[:12]}"
                widget_id = f"wgt_clarify_{turn_id}"
                self._pending_clarification = PendingClarification(
                    turn_id=turn_id,
                    clarification_id=clarification_id,
                    intro=clarifications.intro,
                    questions=list(clarifications.questions),
                    widget_item_id=widget_id,
                )
                self._phase = ResearchPhase.AWAITING_CLARIFICATIONS
                self._message = "Waiting for your clarification answers..."

                await workflow.execute_activity(
                    "publish_progress_activity",
                    PublishProgressInput(
                        thread_id=self._thread_id,
                        turn_id=turn_id,
                        query=query,
                        phase=ResearchPhase.AWAITING_CLARIFICATIONS,
                        message=self._message,
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )
                await workflow.execute_activity(
                    "persist_clarifications_widget_activity",
                    PersistClarificationsWidgetInput(
                        thread_id=self._thread_id,
                        turn_id=turn_id,
                        intro=clarifications.intro,
                        questions=list(clarifications.questions),
                        state="pending",
                    ),
                    result_type=str,
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )

                await workflow.wait_condition(
                    lambda: self._workflow_ended
                    or self._clarification_submission is not None
                )

                if self._workflow_ended:
                    self._pending_clarification = None
                    self._phase = ResearchPhase.INTERRUPTED
                    self._message = "Research workflow ended."
                    return

                submission = self._clarification_submission
                assert submission is not None
                self._pending_clarification = None

                await workflow.execute_activity(
                    "persist_clarifications_widget_activity",
                    PersistClarificationsWidgetInput(
                        thread_id=self._thread_id,
                        turn_id=turn_id,
                        intro=clarifications.intro,
                        questions=list(clarifications.questions),
                        state="skipped" if submission.skipped else "submitted",
                        answers=dict(submission.answers),
                    ),
                    result_type=str,
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )

                self._phase = ResearchPhase.PLANNING
                self._message = "Enhancing research brief from your answers..."
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
                brief = await manager.enhance_query(
                    query,
                    list(clarifications.questions),
                    dict(submission.answers),
                    skipped=submission.skipped,
                    conversation_id=self._openai_conversation_id,
                )
            else:
                # Branch 3: Specific research query already enhanced by InstructionAgent
                assert outcome.brief is not None
                brief = outcome.brief

            # Plan targeted web searches
            self._phase = ResearchPhase.PLANNING
            self._message = "Planning targeted web searches..."
            await workflow.execute_activity(
                "publish_progress_activity",
                PublishProgressInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=brief.topic or query,
                    phase=ResearchPhase.PLANNING,
                    message=self._message,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )

            search_plan = await manager.plan_searches(brief)
            query = search_plan.topic or brief.topic or query
            self._active_query = query
            planned_items = list(search_plan.searches)
            if not planned_items:
                planned_items = [
                    WebSearchItem(query=query, reason="Primary web research query")
                ]
            sub_questions = [item.query for item in planned_items]
            self._active_sub_questions = list(sub_questions)
            selected_depth = "standard"

            # Plan Approval Gate
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
                self._message = "Waiting for your approval on the web search plan..."

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
                    lambda: self._workflow_ended or self._approval_decision is not None
                )

                if self._workflow_ended:
                    self._pending_approval = None
                    self._phase = ResearchPhase.INTERRUPTED
                    self._message = "Research workflow ended."
                    return

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

                reason_by_query = {item.query: item.reason for item in planned_items}
                if decision.sub_questions:
                    sub_questions = list(decision.sub_questions)
                extra = decision.extra_focus.strip()
                if extra and extra not in sub_questions:
                    sub_questions.append(extra)
                planned_items = [
                    WebSearchItem(
                        query=q,
                        reason=reason_by_query.get(
                            q, "User-requested additional focus area"
                        ),
                    )
                    for q in sub_questions
                ]
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

            # Parallel Web Search Fan-Out via SearchAgent (WebSearchTool)
            enhanced_prompt = brief.instructions.strip()
            total = len(planned_items)
            self._phase = ResearchPhase.RESEARCHING
            self._message = f"Searching the web across {total} queries (0/{total})..."
            await workflow.execute_activity(
                "publish_progress_activity",
                PublishProgressInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=query,
                    phase=ResearchPhase.RESEARCHING,
                    message=self._message,
                    enhanced_prompt=enhanced_prompt,
                    sub_questions=sub_questions,
                    completed_sub_questions=0,
                    total_sub_questions=total,
                    update_progress_widget=True,
                    is_new_progress_widget=True,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )

            async def _on_search_complete(
                completed: int, total_count: int, _finding: ResearchFinding | None
            ) -> None:
                self._message = (
                    f"Searching the web ({completed}/{total_count} completed)..."
                )
                await workflow.execute_activity(
                    "publish_progress_activity",
                    PublishProgressInput(
                        thread_id=self._thread_id,
                        turn_id=turn_id,
                        query=query,
                        phase=ResearchPhase.RESEARCHING,
                        message=self._message,
                        enhanced_prompt=enhanced_prompt,
                        sub_questions=sub_questions,
                        completed_sub_questions=completed,
                        total_sub_questions=total_count,
                        update_progress_widget=True,
                        is_new_progress_widget=False,
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_ACTIVITY_RETRY,
                )

            findings = await manager.perform_searches(
                brief,
                planned_items,
                depth=selected_depth,
                on_search_complete=_on_search_complete,
            )

            # Synthesize comprehensive report via WriterAgent
            self._phase = ResearchPhase.SYNTHESIZING
            self._message = "Synthesizing web findings into a deep research report..."
            await workflow.execute_activity(
                "publish_progress_activity",
                PublishProgressInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=query,
                    phase=ResearchPhase.SYNTHESIZING,
                    message=self._message,
                    enhanced_prompt=enhanced_prompt,
                    sub_questions=sub_questions,
                    completed_sub_questions=total,
                    total_sub_questions=total,
                    update_progress_widget=True,
                    is_new_progress_widget=False,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )

            report_data = await manager.write_report(
                brief, findings, depth=selected_depth
            )

            await workflow.execute_activity(
                "publish_report_activity",
                PublishReportInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=query,
                    enhanced_prompt=enhanced_prompt,
                    findings=findings,
                    report=report_data.markdown_report,
                    follow_up_questions=list(report_data.follow_up_questions),
                    conversation_id=self._openai_conversation_id,
                ),
                result_type=str,
                start_to_close_timeout=timedelta(seconds=60),
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
                    enhanced_prompt=enhanced_prompt,
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
        except Exception as exc:  # noqa: BLE001
            if not response_received:
                self._unrecorded_turns.append((query, str(exc)))
                try:
                    await self._flush_unrecorded_turns(best_effort=True)
                except Exception as flush_exc:  # noqa: BLE001
                    # Keep in durable workflow state to flush on the next Turn.
                    workflow.logger.warning(
                        "Best-effort failed-turn flush deferred: %s", flush_exc
                    )
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
