from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.events import InMemoryThreadEventBus, ThreadEvent, ThreadEventKind


@pytest.mark.asyncio
async def test_in_memory_event_bus_replays_and_streams_in_order() -> None:
    bus = InMemoryThreadEventBus()

    seq1 = await bus.publish(
        ThreadEvent(
            thread_id="thr_1",
            turn_id="turn_1",
            kind=ThreadEventKind.PROGRESS,
            payload={"text": "Planning..."},
        )
    )
    seq2 = await bus.publish(
        ThreadEvent(
            thread_id="thr_1",
            turn_id="turn_1",
            kind=ThreadEventKind.TEXT_DELTA,
            payload={"delta": "Hello "},
            activity_id="act_1",
            attempt=1,
        )
    )
    assert seq2 > seq1
    assert await bus.latest_seq("thr_1") == seq2

    received: list[ThreadEvent] = []

    async def _consume() -> None:
        async for ev in bus.subscribe("thr_1", after_seq=seq1):
            received.append(ev)
            if ev.kind == ThreadEventKind.TURN_END:
                break

    consumer = asyncio.create_task(_consume())
    await asyncio.sleep(0.01)

    await bus.publish(
        ThreadEvent(
            thread_id="thr_other",
            turn_id="turn_x",
            kind=ThreadEventKind.PROGRESS,
            payload={"text": "Other thread"},
        )
    )
    await bus.publish(
        ThreadEvent(
            thread_id="thr_1",
            turn_id="turn_1",
            kind=ThreadEventKind.TURN_END,
            payload={"phase": "completed"},
        )
    )

    await asyncio.wait_for(consumer, timeout=1.0)
    assert [ev.kind for ev in received] == [
        ThreadEventKind.TEXT_DELTA,
        ThreadEventKind.TURN_END,
    ]
    assert [ev.seq for ev in received] == [seq2, 4]


@pytest.mark.asyncio
async def test_in_memory_event_bus_prune_by_thread_and_age() -> None:
    bus = InMemoryThreadEventBus()
    old_ts = datetime.now(UTC) - timedelta(hours=30)

    await bus.publish(
        ThreadEvent(
            thread_id="thr_old",
            turn_id="turn_1",
            kind=ThreadEventKind.PROGRESS,
            created_at=old_ts,
        )
    )
    await bus.publish(
        ThreadEvent(
            thread_id="thr_fresh",
            turn_id="turn_1",
            kind=ThreadEventKind.PROGRESS,
        )
    )
    await bus.publish(
        ThreadEvent(
            thread_id="thr_prune_me",
            turn_id="turn_1",
            kind=ThreadEventKind.PROGRESS,
        )
    )

    assert await bus.prune(thread_id="thr_prune_me") == 1
    assert await bus.latest_seq("thr_prune_me") == 0

    assert await bus.prune(older_than_hours=24) == 1
    assert await bus.latest_seq("thr_old") == 0
    assert await bus.latest_seq("thr_fresh") > 0
