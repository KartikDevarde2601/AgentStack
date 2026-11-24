from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool

from app.chatkit_server import DeepResearchChatKitServer
from app.config import Settings, get_settings
from app.db.postgres_store import PostgresChatKitStore
from app.events import InMemoryThreadEventBus, PostgresThreadEventBus, ThreadEventBus
from app.routers import api_router, chatkit_router
from app.temporal.client import TemporalClientManager, TemporalUnavailableError
from app.temporal.orchestrator import TemporalThreadOrchestrator, ThreadOrchestrator

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    store: Any = None,
    orchestrator: ThreadOrchestrator | None = None,
    event_bus: ThreadEventBus | None = None,
    temporal_client_manager: TemporalClientManager | None = None,
) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool: AsyncConnectionPool[Any] | None = None
        active_store = store
        active_orchestrator = orchestrator
        active_event_bus = event_bus
        active_client_manager = temporal_client_manager

        if active_store is None:
            pool = AsyncConnectionPool(
                conninfo=cfg.database_url,
                open=False,
                min_size=1,
                max_size=10,
                timeout=5.0,
            )
            active_store = PostgresChatKitStore(pool)
            try:
                await pool.open(wait=True, timeout=5.0)
                await active_store.initialize()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Postgres initialization failed at startup: %s", exc)

        if active_event_bus is None:
            if pool is not None:
                active_event_bus = PostgresThreadEventBus(
                    pool=pool, dsn=cfg.database_url
                )
            else:
                active_event_bus = InMemoryThreadEventBus()

        if active_orchestrator is None:
            if active_client_manager is None:
                active_client_manager = TemporalClientManager(cfg)
            try:
                await active_client_manager.get_client()
            except TemporalUnavailableError as exc:
                logger.warning("Temporal connection unavailable at startup: %s", exc)
            active_orchestrator = TemporalThreadOrchestrator(
                client_manager=active_client_manager,
                task_queue=cfg.temporal_task_queue,
                model=cfg.openai_model,
            )

        chatkit_server = DeepResearchChatKitServer(
            store=active_store,
            orchestrator=active_orchestrator,
            event_bus=active_event_bus,
        )

        app.state.settings = cfg
        app.state.pool = pool
        app.state.store = active_store
        app.state.event_bus = active_event_bus
        app.state.temporal_client_manager = active_client_manager
        app.state.orchestrator = active_orchestrator
        app.state.chatkit_server = chatkit_server

        try:
            yield
        finally:
            await chatkit_server.close()
            if active_client_manager is not None:
                await active_client_manager.close()
            if pool is not None:
                await pool.close()

    app = FastAPI(title="AgentStack Deep Research Backend", lifespan=lifespan)
    allow_all_origins = "*" in cfg.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[] if allow_all_origins else cfg.cors_origin_list,
        allow_origin_regex=".*" if allow_all_origins else None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_private_network=True,
    )

    app.include_router(chatkit_router)
    app.include_router(api_router)

    return app


app = create_app()
