from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.models.openai_provider import OpenAIProvider
from psycopg_pool import AsyncConnectionPool
from temporalio.worker import Worker

from app.config import Settings, get_settings
from app.db.postgres_store import PostgresChatKitStore
from app.events import PostgresThreadEventBus
from app.research.openai_client import OpenAIResearchClient
from app.temporal.activities import ThreadActivities
from app.temporal.client import TemporalClientManager
from app.temporal.workflows import AgentThreadWorkflow

logger = logging.getLogger(__name__)


async def run_worker(settings: Settings | None = None) -> None:
    cfg = settings or get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    api_key = cfg.require_openai_api_key()
    logger.info(
        "Connecting Temporal worker to %s (namespace=%s, queue=%s, model=%s)",
        cfg.temporal_address,
        cfg.temporal_namespace,
        cfg.temporal_task_queue,
        cfg.openai_model,
    )

    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=cfg.database_url,
        open=False,
        min_size=1,
        max_size=10,
        timeout=5.0,
    )
    await pool.open(wait=True, timeout=5.0)
    client_manager = TemporalClientManager(
        cfg, model_provider=OpenAIProvider(api_key=api_key)
    )
    try:
        store = PostgresChatKitStore(pool)
        await store.initialize()
        event_bus = PostgresThreadEventBus(pool=pool, dsn=cfg.database_url)
        client = await client_manager.get_client()
        research_client = OpenAIResearchClient(
            api_key=api_key,
            model=cfg.openai_model,
        )
        activities = ThreadActivities(
            client=research_client,
            store=store,
            event_bus=event_bus,
        )

        worker = Worker(
            client,
            task_queue=cfg.temporal_task_queue,
            workflows=[AgentThreadWorkflow],
            activities=[
                activities.ensure_openai_conversation_activity,
                activities.record_failed_turns_activity,
                activities.persist_user_message_activity,
                activities.publish_progress_activity,
                activities.persist_clarifications_widget_activity,
                activities.persist_approval_widget_activity,
                activities.publish_report_activity,
                activities.persist_final_item_activity,
                activities.publish_turn_end_activity,
                activities.prune_thread_events_activity,
            ],
        )
        logger.info("Temporal worker started on queue %s", cfg.temporal_task_queue)
        await worker.run()
    finally:
        await client_manager.close()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(run_worker())
