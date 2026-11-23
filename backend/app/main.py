from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from chatkit.server import StreamingResult
from chatkit.store import NotFoundError
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from app.chatkit_server import DeepResearchChatKitServer
from app.config import Settings, get_settings
from app.db.postgres_store import PostgresChatKitStore
from app.domain import RequestContext
from app.events import InMemoryThreadEventBus, PostgresThreadEventBus, ThreadEventBus
from app.temporal.orchestrator import TemporalThreadOrchestrator, ThreadOrchestrator
from app.temporal.worker import connect_temporal_client

logger = logging.getLogger(__name__)


class CreateThreadRequest(BaseModel):
    title: str | None = None


def create_app(
    settings: Settings | None = None,
    store: Any = None,
    orchestrator: ThreadOrchestrator | None = None,
    event_bus: ThreadEventBus | None = None,
) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool: AsyncConnectionPool[Any] | None = None
        active_store = store
        active_orchestrator = orchestrator
        active_event_bus = event_bus
        temporal_client = None

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
            try:
                temporal_client = await connect_temporal_client(cfg)
                active_orchestrator = TemporalThreadOrchestrator(
                    client=temporal_client,
                    task_queue=cfg.temporal_task_queue,
                    model=cfg.openai_model,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Temporal connection unavailable at startup: %s", exc)

        chatkit_server = DeepResearchChatKitServer(
            store=active_store,
            orchestrator=active_orchestrator,
            event_bus=active_event_bus,
        )

        app.state.settings = cfg
        app.state.pool = pool
        app.state.store = active_store
        app.state.event_bus = active_event_bus
        app.state.temporal_client = temporal_client
        app.state.orchestrator = active_orchestrator
        app.state.chatkit_server = chatkit_server

        try:
            yield
        finally:
            await chatkit_server.close()
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

    async def _ensure_orchestrator(request: Request) -> ThreadOrchestrator | None:
        if request.app.state.orchestrator is not None:
            return request.app.state.orchestrator
        try:
            client = await connect_temporal_client(request.app.state.settings)
            orch = TemporalThreadOrchestrator(
                client=client,
                task_queue=request.app.state.settings.temporal_task_queue,
                model=request.app.state.settings.openai_model,
            )
            request.app.state.temporal_client = client
            request.app.state.orchestrator = orch
            request.app.state.chatkit_server.set_orchestrator(orch)
            return orch
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lazy Temporal reconnect failed: %s", exc)
            return None

    @app.post("/chatkit")
    async def chatkit_endpoint(request: Request) -> Response:
        await _ensure_orchestrator(request)
        payload = await request.body()
        context = RequestContext()
        result = await request.app.state.chatkit_server.process(payload, context)
        if isinstance(result, StreamingResult):
            return StreamingResponse(result, media_type="text/event-stream")
        return Response(content=result.json, media_type="application/json")

    @app.get("/api/threads")
    async def list_threads(request: Request) -> dict[str, Any]:
        threads = await request.app.state.store.list_thread_summaries()
        return {"threads": [t.model_dump(mode="json") for t in threads]}

    @app.post("/api/threads")
    async def create_thread(
        request: Request, body: CreateThreadRequest | None = None
    ) -> dict[str, Any]:
        title = body.title if body is not None else None
        summary = await request.app.state.store.create_thread(
            title=title, context=RequestContext()
        )
        return summary.model_dump(mode="json")

    @app.delete("/api/threads/{thread_id}")
    async def delete_thread(thread_id: str, request: Request) -> dict[str, Any]:
        ctx = RequestContext()
        try:
            await request.app.state.store.load_thread(thread_id, ctx)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await request.app.state.store.delete_thread(thread_id, ctx)
        orch = request.app.state.orchestrator
        if orch is not None:
            try:
                await orch.end_workflow(thread_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to signal end_workflow for %s: %s", thread_id, exc)
        return {"deleted": True, "id": thread_id}

    @app.get("/api/threads/{thread_id}/runs")
    async def list_thread_runs(thread_id: str, request: Request) -> dict[str, Any]:
        runs = await request.app.state.store.list_research_runs(thread_id)
        return {"runs": [r.model_dump(mode="json") for r in runs]}

    @app.get("/api/threads/{thread_id}/state")
    async def get_thread_state(thread_id: str, request: Request) -> dict[str, Any]:
        orch = await _ensure_orchestrator(request)
        if orch is None:
            raise HTTPException(
                status_code=503, detail="Temporal orchestrator unavailable"
            )
        state = await orch.get_state(thread_id)
        return state.model_dump(mode="json")

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        postgres_ok = await request.app.state.store.ping()
        temporal_ok = request.app.state.orchestrator is not None
        if not temporal_ok:
            await _ensure_orchestrator(request)
            temporal_ok = request.app.state.orchestrator is not None
        return {
            "status": "ok",
            "postgres": postgres_ok,
            "temporal": temporal_ok,
            "openai_mode": request.app.state.settings.openai_mode,
        }

    return app


app = create_app()
