from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.events.bus import ThreadEvent, ThreadEventKind

_NOTIFY_CHANNEL = "thread_events"


class PostgresThreadEventBus:
    """Postgres-backed ThreadEventBus using run_events + LISTEN/NOTIFY."""

    def __init__(self, pool: AsyncConnectionPool[Any], dsn: str) -> None:
        self._pool = pool
        self._dsn = dsn

    async def publish(self, event: ThreadEvent) -> int:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    INSERT INTO run_events (
                        thread_id, turn_id, kind, payload, activity_id, attempt, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING seq
                    """,
                    (
                        event.thread_id,
                        event.turn_id,
                        event.kind.value,
                        Jsonb(event.payload),
                        event.activity_id,
                        event.attempt,
                        event.created_at,
                    ),
                )
                row = await cur.fetchone()
                assert row is not None
                seq = int(row["seq"])
                await cur.execute(
                    "SELECT pg_notify(%s, %s)",
                    (_NOTIFY_CHANNEL, event.thread_id),
                )
            await conn.commit()
        return seq

    async def _fetch_after(self, thread_id: str, after_seq: int) -> list[ThreadEvent]:
        async with (
            self._pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            await cur.execute(
                """
                SELECT seq, thread_id, turn_id, kind, payload, activity_id, attempt, created_at
                FROM run_events
                WHERE thread_id = %s AND seq > %s
                ORDER BY seq ASC
                """,
                (thread_id, after_seq),
            )
            rows = await cur.fetchall()
        return [
            ThreadEvent(
                seq=int(r["seq"]),
                thread_id=r["thread_id"],
                turn_id=r["turn_id"],
                kind=ThreadEventKind(r["kind"]),
                payload=dict(r["payload"] or {}),
                activity_id=r["activity_id"],
                attempt=int(r["attempt"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def subscribe(
        self, thread_id: str, *, after_seq: int = 0
    ) -> AsyncIterator[ThreadEvent]:
        last_seq = after_seq
        async with await psycopg.AsyncConnection.connect(
            self._dsn, autocommit=True
        ) as listen_conn:
            await listen_conn.execute(f"LISTEN {_NOTIFY_CHANNEL}")

            for ev in await self._fetch_after(thread_id, last_seq):
                last_seq = ev.seq
                yield ev

            while True:
                async for notify in listen_conn.notifies(timeout=5.0):
                    if notify.payload == thread_id:
                        break

                events = await self._fetch_after(thread_id, last_seq)
                for ev in events:
                    last_seq = ev.seq
                    yield ev

    async def latest_seq(self, thread_id: str) -> int:
        async with (
            self._pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            await cur.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM run_events WHERE thread_id = %s",
                (thread_id,),
            )
            row = await cur.fetchone()
        return int(row["max_seq"]) if row is not None else 0

    async def prune(
        self, *, thread_id: str | None = None, older_than_hours: int = 24
    ) -> int:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                if thread_id is not None:
                    await cur.execute(
                        "DELETE FROM run_events WHERE thread_id = %s",
                        (thread_id,),
                    )
                else:
                    await cur.execute(
                        "DELETE FROM run_events WHERE created_at < NOW() - make_interval(hours => %s)",
                        (older_than_hours,),
                    )
                deleted = cur.rowcount or 0
            await conn.commit()
        return deleted
