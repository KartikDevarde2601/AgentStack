from __future__ import annotations

from chatkit.server import StreamingResult
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from app.domain import RequestContext

router = APIRouter(tags=["chatkit"])


@router.post("/chatkit")
async def chatkit_endpoint(request: Request) -> Response:
    payload = await request.body()
    context = RequestContext()
    result = await request.app.state.chatkit_server.process(payload, context)
    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    return Response(content=result.json, media_type="application/json")
