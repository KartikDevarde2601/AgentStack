from __future__ import annotations

import asyncio
import uuid

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from app.domain import (
    ApprovalDecision,
    ResearchFinding,
    ResearchPhase,
    ResearchPlan,
    ResearchProgress,
)
from app.research.openai_client import OpenAIResearchClient
from app.temporal.activities import (
    GatherFindingInput,
    ResearchActivities,
    SynthesizeReportInput,
)
from app.temporal.workflows import DeepResearchWorkflow


@pytest.mark.asyncio
async def test_deep_research_workflow_end_to_end() -> None:
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        task_queue = f"test-research-{uuid.uuid4().hex[:8]}"
        activities = ResearchActivities(OpenAIResearchClient(api_key=""))

        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[DeepResearchWorkflow],
            activities=[
                activities.plan_research_activity,
                activities.gather_finding_activity,
                activities.synthesize_report_activity,
            ],
        ):
            workflow_id = f"wf-{uuid.uuid4().hex[:8]}"
            query = "How does Temporal guarantee durable execution?"
            handle = await env.client.start_workflow(
                DeepResearchWorkflow.run,
                query,
                id=workflow_id,
                task_queue=task_queue,
            )

            result: ResearchProgress = await handle.result()
            queried: ResearchProgress = await handle.query(
                DeepResearchWorkflow.get_progress
            )

            assert result.workflow_id == workflow_id
            assert result.query == query
            assert result.phase == ResearchPhase.COMPLETED
            assert len(result.sub_questions) == 3
            assert result.completed_sub_questions == 3
            assert result.total_sub_questions == 3
            assert result.error is None
            assert result.report is not None
            assert f"# Deep Research Report: {query}" in result.report
            assert "## Executive Summary" in result.report
            assert "## Sources" in result.report

            assert queried.phase == ResearchPhase.COMPLETED
            assert queried.report == result.report


@pytest.mark.asyncio
async def test_deep_research_workflow_phase_transitions() -> None:
    observed_phases: list[ResearchPhase] = []
    release_plan = asyncio.Event()
    in_plan = asyncio.Event()
    release_research = asyncio.Event()
    in_research = asyncio.Event()
    release_synth = asyncio.Event()
    in_synth = asyncio.Event()

    demo_client = OpenAIResearchClient(api_key="")

    @activity.defn(name="plan_research_activity")
    async def gated_plan(query: str) -> ResearchPlan:
        in_plan.set()
        await release_plan.wait()
        return await demo_client.generate_research_plan(query)

    @activity.defn(name="gather_finding_activity")
    async def gated_gather(params: GatherFindingInput) -> ResearchFinding:
        in_research.set()
        await release_research.wait()
        return await demo_client.investigate_sub_question(
            params.query, params.sub_question
        )

    @activity.defn(name="synthesize_report_activity")
    async def gated_synth(params: SynthesizeReportInput) -> str:
        in_synth.set()
        await release_synth.wait()
        return await demo_client.synthesize_report(params.query, params.findings)

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        task_queue = f"test-phases-{uuid.uuid4().hex[:8]}"
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[DeepResearchWorkflow],
            activities=[gated_plan, gated_gather, gated_synth],
        ):
            handle = await env.client.start_workflow(
                DeepResearchWorkflow.run,
                "Event-driven architectures",
                id=f"wf-phases-{uuid.uuid4().hex[:8]}",
                task_queue=task_queue,
            )

            await in_plan.wait()
            observed_phases.append(
                (await handle.query(DeepResearchWorkflow.get_progress)).phase
            )
            release_plan.set()

            await in_research.wait()
            observed_phases.append(
                (await handle.query(DeepResearchWorkflow.get_progress)).phase
            )
            release_research.set()

            await in_synth.wait()
            observed_phases.append(
                (await handle.query(DeepResearchWorkflow.get_progress)).phase
            )
            release_synth.set()

            final = await handle.result()
            observed_phases.append(final.phase)

            assert observed_phases == [
                ResearchPhase.PLANNING,
                ResearchPhase.RESEARCHING,
                ResearchPhase.SYNTHESIZING,
                ResearchPhase.COMPLETED,
            ]


@pytest.mark.asyncio
async def test_deep_research_workflow_handles_activity_failure() -> None:
    @activity.defn(name="plan_research_activity")
    async def failing_plan(query: str) -> ResearchPlan:
        raise RuntimeError("planner exploded")

    @activity.defn(name="gather_finding_activity")
    async def noop_gather(params: GatherFindingInput) -> ResearchFinding:
        raise AssertionError("should not run")

    @activity.defn(name="synthesize_report_activity")
    async def noop_synth(params: SynthesizeReportInput) -> str:
        raise AssertionError("should not run")

    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        task_queue = f"test-fail-{uuid.uuid4().hex[:8]}"
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[DeepResearchWorkflow],
            activities=[failing_plan, noop_gather, noop_synth],
        ):
            handle = await env.client.start_workflow(
                DeepResearchWorkflow.run,
                "Broken query",
                id=f"wf-fail-{uuid.uuid4().hex[:8]}",
                task_queue=task_queue,
            )
            result: ResearchProgress = await handle.result()
            assert result.phase == ResearchPhase.FAILED
            assert result.error is not None
            assert result.report is None


@pytest.mark.asyncio
async def test_deep_research_workflow_hitl_approval_and_rejection() -> None:
    async with await WorkflowEnvironment.start_time_skipping(
        data_converter=pydantic_data_converter
    ) as env:
        task_queue = f"test-hitl-{uuid.uuid4().hex[:8]}"
        activities = ResearchActivities(OpenAIResearchClient(api_key=""))

        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[DeepResearchWorkflow],
            activities=[
                activities.plan_research_activity,
                activities.gather_finding_activity,
                activities.synthesize_report_activity,
            ],
        ):
            # 1. Approve flow with edited sub-questions + extra focus
            approve_id = f"wf-hitl-ok-{uuid.uuid4().hex[:8]}"
            handle = await env.client.start_workflow(
                DeepResearchWorkflow.run,
                args=["Postgres vs SQLite", True],
                id=approve_id,
                task_queue=task_queue,
            )

            progress = await handle.query(DeepResearchWorkflow.get_progress)
            for _ in range(50):
                if progress.phase == ResearchPhase.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.02)
                progress = await handle.query(DeepResearchWorkflow.get_progress)

            assert progress.phase == ResearchPhase.AWAITING_APPROVAL
            assert len(progress.sub_questions) == 3

            await handle.signal(
                DeepResearchWorkflow.submit_approval,
                ApprovalDecision(
                    approved=True,
                    sub_questions=progress.sub_questions[:1],
                    depth="deep",
                    extra_focus="Operational backup and restore tradeoffs",
                ),
            )

            result = await handle.result()
            assert result.phase == ResearchPhase.COMPLETED
            assert len(result.sub_questions) == 2
            assert "Operational backup and restore tradeoffs" in result.sub_questions
            assert len(result.findings) == 2
            assert result.report is not None

            # 2. Reject flow cancels cleanly without running research activities
            reject_id = f"wf-hitl-no-{uuid.uuid4().hex[:8]}"
            reject_handle = await env.client.start_workflow(
                DeepResearchWorkflow.run,
                args=["Rejected topic", True],
                id=reject_id,
                task_queue=task_queue,
            )

            rej_progress = await reject_handle.query(DeepResearchWorkflow.get_progress)
            for _ in range(50):
                if rej_progress.phase == ResearchPhase.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.02)
                rej_progress = await reject_handle.query(
                    DeepResearchWorkflow.get_progress
                )
            assert rej_progress.phase == ResearchPhase.AWAITING_APPROVAL

            await reject_handle.signal(
                DeepResearchWorkflow.submit_approval,
                ApprovalDecision(approved=False, reason="Not needed"),
            )

            rejected = await reject_handle.result()
            assert rejected.phase == ResearchPhase.COMPLETED
            assert rejected.approval_decision is not None
            assert rejected.approval_decision.approved is False
            assert rejected.findings == []
            assert rejected.report is not None
            assert "Research Cancelled" in rejected.report
