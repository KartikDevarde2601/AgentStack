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

from app.chatkit_server import (
    DeepResearchChatKitServer,
    ResearchWorkflowRunner,
    TemporalResearchRunner,
)
from app.config import Settings, get_settings
from app.db.postgres_store import PostgresChatKitStore
from app.domain import RequestContext
from app.temporal.worker import connect_temporal_client

logger = logging.getLogger(__name__)


class CreateThreadRequest(BaseModel):
    title: str | None = None


def create_app(
    settings: Settings | None = None,
    store: Any = None,
    runner: ResearchWorkflowRunner | None = None,
    poll_interval_seconds: float = 0.25,
) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool: AsyncConnectionPool[Any] | None = None
        active_store = store
        active_runner = runner
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
            except Exception as exc:
                logger.warning("Postgres initialization failed at startup: %s", exc)

        if active_runner is None:
            try:
                temporal_client = await connect_temporal_client(cfg)
                active_runner = TemporalResearchRunner(
                    client=temporal_client,
                    task_queue=cfg.temporal_task_queue,
                )
            except Exception as exc:
                logger.warning("Temporal connection unavailable at startup: %s", exc)

        chatkit_server = DeepResearchChatKitServer(
            store=active_store,
            runner=active_runner,
            run_store=active_store,
            poll_interval_seconds=poll_interval_seconds,
        )

        app.state.settings = cfg
        app.state.pool = pool
        app.state.store = active_store
        app.state.temporal_client = temporal_client
        app.state.runner = active_runner
        app.state.chatkit_server = chatkit_server

        try:
            yield
        finally:
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
    )

    @app.middleware("http")
    async def allow_private_network_access(request: Request, call_next):
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network") == "true":
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    async def _ensure_temporal_runner(request: Request) -> None:
        if request.app.state.runner is not None:
            return
        try:
            client = await connect_temporal_client(request.app.state.settings)
            temporal_runner = TemporalResearchRunner(
                client=client,
                task_queue=request.app.state.settings.temporal_task_queue,
            )
            request.app.state.temporal_client = client
            request.app.state.runner = temporal_runner
            request.app.state.chatkit_server.set_runner(temporal_runner)
        except Exception as exc:
            logger.warning("Lazy Temporal reconnect failed: %s", exc)

    @app.post("/chatkit")
    async def chatkit_endpoint(request: Request) -> Response:
        await _ensure_temporal_runner(request)
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
        return {"deleted": True, "id": thread_id}

    @app.get("/api/threads/{thread_id}/runs")
    async def list_thread_runs(thread_id: str, request: Request) -> dict[str, Any]:
        runs = await request.app.state.store.list_research_runs(thread_id)
        return {"runs": [r.model_dump(mode="json") for r in runs]}

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        postgres_ok = await request.app.state.store.ping()
        temporal_ok = request.app.state.runner is not None
        if not temporal_ok:
            await _ensure_temporal_runner(request)
            temporal_ok = request.app.state.runner is not None
        return {
            "status": "ok",
            "postgres": postgres_ok,
            "temporal": temporal_ok,
            "openai_mode": request.app.state.settings.openai_mode,
        }

    return app


app = create_app()
