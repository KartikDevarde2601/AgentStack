from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from app.events.bus import ThreadEvent


class InMemoryThreadEventBus:
    """In-memory ThreadEventBus adapter for pytest."""

    def __init__(self) -> None:
        self._seq = 0
        self._events_by_thread: dict[str, list[ThreadEvent]] = defaultdict(list)
        self._subscribers: dict[str, set[asyncio.Queue[ThreadEvent]]] = defaultdict(set)

    async def publish(self, event: ThreadEvent) -> int:
        self._seq += 1
        stored = event.model_copy(update={"seq": self._seq})
        self._events_by_thread[stored.thread_id].append(stored)
        for queue in list(self._subscribers.get(stored.thread_id, ())):
            queue.put_nowait(stored)
        return self._seq

    async def subscribe(
        self, thread_id: str, *, after_seq: int = 0
    ) -> AsyncIterator[ThreadEvent]:
        queue: asyncio.Queue[ThreadEvent] = asyncio.Queue()
        self._subscribers[thread_id].add(queue)
        last_yielded = after_seq
        try:
            for existing in list(self._events_by_thread.get(thread_id, ())):
                if existing.seq > last_yielded:
                    last_yielded = existing.seq
                    yield existing
            while True:
                event = await queue.get()
                if event.seq > last_yielded:
                    last_yielded = event.seq
                    yield event
        finally:
            self._subscribers[thread_id].discard(queue)

    async def latest_seq(self, thread_id: str) -> int:
        events = self._events_by_thread.get(thread_id)
        if not events:
            return 0
        return events[-1].seq

    async def prune(
        self, *, thread_id: str | None = None, older_than_hours: int = 24
    ) -> int:
        cutoff = datetime.now(UTC) - timedelta(hours=older_than_hours)
        removed = 0
        if thread_id is not None:
            removed = len(self._events_by_thread.pop(thread_id, []))
            return removed

        for tid in list(self._events_by_thread.keys()):
            before = self._events_by_thread[tid]
            kept = [ev for ev in before if ev.created_at >= cutoff]
            removed += len(before) - len(kept)
            if kept:
                self._events_by_thread[tid] = kept
            else:
                self._events_by_thread.pop(tid, None)
        return removed
