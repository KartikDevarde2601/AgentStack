from __future__ import annotations

import asyncio
import logging

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from app.config import Settings, get_settings
from app.research.openai_client import OpenAIResearchClient
from app.temporal.activities import ResearchActivities
from app.temporal.workflows import DeepResearchWorkflow

logger = logging.getLogger(__name__)


async def connect_temporal_client(settings: Settings) -> Client:
    return await Client.connect(
        settings.temporal_address,
        namespace=settings.temporal_namespace,
        data_converter=pydantic_data_converter,
    )


async def run_worker(settings: Settings | None = None) -> None:
    cfg = settings or get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "Connecting Temporal worker to %s (namespace=%s, queue=%s, openai_mode=%s)",
        cfg.temporal_address,
        cfg.temporal_namespace,
        cfg.temporal_task_queue,
        cfg.openai_mode,
    )
    client = await connect_temporal_client(cfg)
    research_client = OpenAIResearchClient(
        api_key=cfg.openai_api_key,
        model=cfg.openai_model,
    )
    activities = ResearchActivities(research_client)

    worker = Worker(
        client,
        task_queue=cfg.temporal_task_queue,
        workflows=[DeepResearchWorkflow],
        activities=[
            activities.plan_research_activity,
            activities.gather_finding_activity,
            activities.synthesize_report_activity,
        ],
    )
    logger.info("Temporal worker started on queue %s", cfg.temporal_task_queue)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
