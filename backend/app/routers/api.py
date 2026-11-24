from __future__ import annotations

import logging
from typing import Any

from chatkit.store import NotFoundError
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.domain import RequestContext
from app.temporal.client import TemporalUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["api"])


class CreateThreadRequest(BaseModel):
    title: str | None = None


@router.get("/api/threads")
async def list_threads(request: Request) -> dict[str, Any]:
    threads = await request.app.state.store.list_thread_summaries()
    return {"threads": [t.model_dump(mode="json") for t in threads]}


@router.post("/api/threads")
async def create_thread(
    request: Request, body: CreateThreadRequest | None = None
) -> dict[str, Any]:
    title = body.title if body is not None else None
    summary = await request.app.state.store.create_thread(
        title=title, context=RequestContext()
    )
    return summary.model_dump(mode="json")


@router.delete("/api/threads/{thread_id}")
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


@router.get("/api/threads/{thread_id}/runs")
async def list_thread_runs(thread_id: str, request: Request) -> dict[str, Any]:
    runs = await request.app.state.store.list_research_runs(thread_id)
    return {"runs": [r.model_dump(mode="json") for r in runs]}


@router.get("/api/threads/{thread_id}/state")
async def get_thread_state(thread_id: str, request: Request) -> dict[str, Any]:
    try:
        state = await request.app.state.orchestrator.get_state(thread_id)
    except TemporalUnavailableError as exc:
        raise HTTPException(
            status_code=503, detail="Temporal orchestrator unavailable"
        ) from exc
    return state.model_dump(mode="json")


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    postgres_ok = await request.app.state.store.ping()
    manager = request.app.state.temporal_client_manager
    if manager is not None:
        temporal_ok = await manager.ping()
    else:
        temporal_ok = request.app.state.orchestrator is not None
    return {
        "status": "ok",
        "postgres": postgres_ok,
        "temporal": temporal_ok,
        "openai_mode": request.app.state.settings.openai_mode,
    }
