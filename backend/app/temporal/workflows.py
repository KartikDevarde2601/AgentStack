from __future__ import annotations

import asyncio
from datetime import timedelta

from pydantic import BaseModel, Field
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from app.domain import (
        ApprovalDecision,
        IncomingMessage,
        PendingApproval,
        QueuedMessage,
        ResearchFinding,
        ResearchPhase,
        ResearchPlan,
        ThreadWorkflowState,
    )
    from app.temporal.activities import (
        CloseToolCallsInput,
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
    queued_messages: list[QueuedMessage] = Field(default_factory=list)
    unrecorded_turns: list[tuple[str, str]] = Field(default_factory=list)
    require_approval: bool = True
    idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS
    max_turns_before_continue: int = DEFAULT_MAX_TURNS_BEFORE_CONTINUE


@workflow.defn(name="AgentThreadWorkflow")
class AgentThreadWorkflow:
    """Durable per-Thread Entity Workflow (id = thread-{thread_id}).

    Owns turn serialization, FIFO Queued Messages, Approval Gate suspension,
    mid-Turn Steer interruption with strict call_id closure, 30-min idle exit,
    and continue_as_new history bounds.
    """

    def __init__(self) -> None:
        self._thread_id: str = ""
        self._openai_conversation_id: str | None = None
        self._phase: ResearchPhase = ResearchPhase.IDLE
        self._active_turn_id: str | None = None
        self._message: str = ""
        self._queue: list[QueuedMessage] = []
        self._pending_approval: PendingApproval | None = None
        self._approval_decision: ApprovalDecision | None = None
        self._open_call_ids: list[str] = []
        self._active_turn_task: asyncio.Task[None] | None = None
        self._steer_message: QueuedMessage | None = None
        self._steer_requested: bool = False
        self._turns_since_continue: int = 0
        self._active_query: str = ""
        self._active_sub_questions: list[str] = []
        self._active_findings: list[ResearchFinding] = []
        self._unrecorded_turns: list[tuple[str, str]] = []

    def _to_queued_message(self, msg: IncomingMessage) -> QueuedMessage:
        return QueuedMessage(
            message_id=msg.message_id,
            text=msg.text,
            created_at=workflow.now(),
            already_persisted=msg.already_persisted,
        )

    @workflow.signal
    def enqueue_message(self, msg: IncomingMessage) -> None:
        queued = self._to_queued_message(msg)
        # If parked at an Approval Gate and a new message arrives, treat it as a
        # Steer so the stale approval widget is cancelled and open tool calls are closed.
        if self._pending_approval is not None or msg.mode == "steer":
            self._request_steer(queued)
            return
        self._queue.append(queued)

    @workflow.signal
    def steer(self, msg: IncomingMessage) -> None:
        self._request_steer(self._to_queued_message(msg))

    def _request_steer(self, queued: QueuedMessage) -> None:
        if self._active_turn_task is not None and not self._active_turn_task.done():
            self._steer_message = queued
            self._steer_requested = True
            self._active_turn_task.cancel()
        else:
            self._queue.insert(0, queued)

    @workflow.signal
    def cancel_queued(self, message_id: str) -> None:
        self._queue = [m for m in self._queue if m.message_id != message_id]

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
            queued_messages=list(self._queue),
            pending_approval=self._pending_approval,
            turns_since_continue=self._turns_since_continue,
        )

    @workflow.run
    async def run(self, params: AgentThreadInput) -> None:
        self._thread_id = params.thread_id
        self._openai_conversation_id = params.openai_conversation_id
        self._queue.extend(params.queued_messages)
        self._unrecorded_turns.extend(params.unrecorded_turns)

        while True:
            if not self._queue:
                self._phase = ResearchPhase.IDLE
                self._active_turn_id = None
                self._message = ""
                try:
                    await workflow.wait_condition(
                        lambda: bool(self._queue),
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

            next_msg = self._queue.pop(0)
            turn_id = f"turn_{workflow.uuid4().hex[:10]}"
            self._steer_requested = False
            self._steer_message = None
            self._approval_decision = None
            self._pending_approval = None
            self._open_call_ids = []
            self._active_turn_id = turn_id
            self._active_query = next_msg.text
            self._active_sub_questions = []
            self._active_findings = []

            self._active_turn_task = asyncio.create_task(
                self._run_turn(
                    msg=next_msg,
                    turn_id=turn_id,
                    require_approval=params.require_approval,
                )
            )
            try:
                await self._active_turn_task
            except asyncio.CancelledError:
                if not self._steer_requested:
                    raise
                await self._handle_steer_cleanup(turn_id=turn_id, query=next_msg.text)
                if self._steer_message is not None:
                    self._queue.insert(0, self._steer_message)
                    self._steer_message = None
                self._steer_requested = False
            finally:
                self._active_turn_task = None
                self._pending_approval = None
                self._open_call_ids = []

            self._turns_since_continue += 1
            if (
                workflow.info().is_continue_as_new_suggested()
                or self._turns_since_continue >= params.max_turns_before_continue
            ):
                workflow.continue_as_new(
                    AgentThreadInput(
                        thread_id=self._thread_id,
                        openai_conversation_id=self._openai_conversation_id,
                        queued_messages=list(self._queue),
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
        msg: QueuedMessage,
        turn_id: str,
        require_approval: bool,
    ) -> None:
        query = msg.text
        plan_succeeded = False
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

            # Flush any earlier failed turns that OpenAI's transactional
            # responses.create rejected before appending to the Conversation.
            await self._flush_unrecorded_turns()

            plan: ResearchPlan = await workflow.execute_activity(
                "plan_research_activity",
                PlanResearchInput(
                    query=query,
                    conversation_id=self._openai_conversation_id,
                ),
                result_type=ResearchPlan,
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=_ACTIVITY_RETRY,
            )
            plan_succeeded = True
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
                    lambda: self._approval_decision is not None or self._steer_requested
                )
                if self._steer_requested:
                    raise asyncio.CancelledError("Steered while awaiting approval")

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
                        "persist_interrupted_or_final_activity",
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
                self._active_findings.append(finding)
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
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not plan_succeeded:
                self._unrecorded_turns.append((query, str(exc)))
                try:
                    await self._flush_unrecorded_turns(best_effort=True)
                except Exception:
                    # Keep in durable workflow state to flush on the next Turn.
                    pass
            self._phase = ResearchPhase.FAILED
            self._message = f"Deep research failed: {exc}"
            await workflow.execute_activity(
                "persist_interrupted_or_final_activity",
                PersistFinalItemInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    query=query,
                    phase=ResearchPhase.FAILED,
                    message=self._message,
                    sub_questions=self._active_sub_questions,
                    text=(
                        "### Research Failed\n\n"
                        f"Unable to complete research for **{query}**.\n\n"
                        f"Error: `{exc}`"
                    ),
                    error=str(exc),
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )

    async def _handle_steer_cleanup(self, *, turn_id: str, query: str) -> None:
        """Enforce strict call_id closure and preserve partial output as Interrupted."""
        if self._pending_approval is not None:
            await workflow.execute_activity(
                "persist_approval_widget_activity",
                PersistApprovalWidgetInput(
                    thread_id=self._thread_id,
                    turn_id=turn_id,
                    sub_questions=self._pending_approval.sub_questions,
                    state="cancelled",
                ),
                result_type=str,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )

        if self._open_call_ids and self._openai_conversation_id:
            await workflow.execute_activity(
                "close_dangling_tool_calls_activity",
                CloseToolCallsInput(
                    conversation_id=self._openai_conversation_id,
                    call_ids=list(self._open_call_ids),
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_ACTIVITY_RETRY,
            )

        partial_lines = [
            f"### Research Interrupted — {query}",
            "",
            "_Interrupted by a new steering instruction._",
        ]
        if self._active_findings:
            partial_lines.extend(["", "**Partial findings preserved:**", ""])
            for finding in self._active_findings:
                partial_lines.append(f"- **{finding.sub_question}**: {finding.summary}")

        self._phase = ResearchPhase.INTERRUPTED
        self._message = "Interrupted by user steer."
        await workflow.execute_activity(
            "persist_interrupted_or_final_activity",
            PersistFinalItemInput(
                thread_id=self._thread_id,
                turn_id=turn_id,
                query=query,
                phase=ResearchPhase.INTERRUPTED,
                message=self._message,
                sub_questions=self._active_sub_questions,
                text="\n".join(partial_lines),
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_ACTIVITY_RETRY,
        )
