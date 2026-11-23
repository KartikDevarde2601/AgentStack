from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ThreadEventKind(StrEnum):
    PROGRESS = "progress"
    ITEM_ADDED = "item_added"
    TEXT_DELTA = "text_delta"
    ITEM_DONE = "item_done"
    ITEM_REPLACED = "item_replaced"
    APPROVAL_PAUSE = "approval_pause"
    TURN_END = "turn_end"


class ThreadEvent(BaseModel):
    seq: int = 0
    thread_id: str
    turn_id: str
    kind: ThreadEventKind
    payload: dict[str, Any] = Field(default_factory=dict)
    activity_id: str | None = None
    attempt: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ThreadEventBus(Protocol):
    async def publish(self, event: ThreadEvent) -> int: ...

    def subscribe(
        self, thread_id: str, *, after_seq: int = 0
    ) -> AsyncIterator[ThreadEvent]: ...

    async def latest_seq(self, thread_id: str) -> int: ...

    async def prune(
        self, *, thread_id: str | None = None, older_than_hours: int = 24
    ) -> int: ...
