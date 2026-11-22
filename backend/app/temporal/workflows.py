from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from app.domain import ResearchFinding, ResearchPhase, ResearchPlan, ResearchProgress
    from app.temporal.activities import GatherFindingInput, SynthesizeReportInput

_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_attempts=3,
)


@workflow.defn(name="DeepResearchWorkflow")
class DeepResearchWorkflow:
    def __init__(self) -> None:
        self._progress = ResearchProgress(
            workflow_id="",
            query="",
            phase=ResearchPhase.QUEUED,
            message="Queued deep research workflow...",
        )

    @workflow.query
    def get_progress(self) -> ResearchProgress:
        return self._progress

    @workflow.run
    async def run(self, query: str) -> ResearchProgress:
        workflow_id = workflow.info().workflow_id
        self._progress = ResearchProgress(
            workflow_id=workflow_id,
            query=query,
            phase=ResearchPhase.PLANNING,
            message="Planning research sub-questions...",
        )

        try:
            plan: ResearchPlan = await workflow.execute_activity(
                "plan_research_activity",
                query,
                result_type=ResearchPlan,
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=_ACTIVITY_RETRY,
            )

            sub_questions = plan.sub_questions
            total = len(sub_questions)
            self._progress = ResearchProgress(
                workflow_id=workflow_id,
                query=query,
                phase=ResearchPhase.RESEARCHING,
                message=f"Researching {total} sub-questions in parallel (0/{total})...",
                sub_questions=sub_questions,
                completed_sub_questions=0,
                total_sub_questions=total,
            )

            async def _investigate(sub_question: str) -> ResearchFinding:
                finding: ResearchFinding = await workflow.execute_activity(
                    "gather_finding_activity",
                    GatherFindingInput(query=query, sub_question=sub_question),
                    result_type=ResearchFinding,
                    start_to_close_timeout=timedelta(seconds=90),
                    retry_policy=_ACTIVITY_RETRY,
                )
                completed = self._progress.completed_sub_questions + 1
                self._progress = self._progress.model_copy(
                    update={
                        "completed_sub_questions": completed,
                        "message": f"Researching sub-questions ({completed}/{total} completed)...",
                    }
                )
                return finding

            findings = list(
                await asyncio.gather(*(_investigate(q) for q in sub_questions))
            )

            self._progress = self._progress.model_copy(
                update={
                    "phase": ResearchPhase.SYNTHESIZING,
                    "message": "Synthesizing findings into a structured report...",
                }
            )

            report: str = await workflow.execute_activity(
                "synthesize_report_activity",
                SynthesizeReportInput(query=query, findings=findings),
                result_type=str,
                start_to_close_timeout=timedelta(seconds=90),
                retry_policy=_ACTIVITY_RETRY,
            )

            self._progress = self._progress.model_copy(
                update={
                    "phase": ResearchPhase.COMPLETED,
                    "message": "Deep research completed.",
                    "report": report,
                }
            )
            return self._progress
        except Exception as exc:
            self._progress = self._progress.model_copy(
                update={
                    "phase": ResearchPhase.FAILED,
                    "message": f"Deep research failed: {exc}",
                    "error": str(exc),
                }
            )
            return self._progress
