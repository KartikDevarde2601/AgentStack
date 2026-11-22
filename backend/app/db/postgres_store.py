from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chatkit.store import NotFoundError, Store
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import TypeAdapter

from app.domain import (
    RequestContext,
    ResearchPhase,
    ResearchRunRecord,
    ThreadSummary,
)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_THREAD_ITEM_ADAPTER = TypeAdapter(ThreadItem)
_ATTACHMENT_ADAPTER = TypeAdapter(Attachment)


def _normalize_order(order: str) -> str:
    return "ASC" if order.lower() == "asc" else "DESC"


class PostgresChatKitStore(Store[RequestContext]):
    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    async def initialize(self) -> None:
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(schema_sql)
            await conn.commit()

    async def ping(self) -> bool:
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    await cur.fetchone()
            return True
        except Exception:
            return False

    async def load_thread(
        self, thread_id: str, context: RequestContext
    ) -> ThreadMetadata:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT payload FROM chatkit_threads WHERE id = %s",
                    (thread_id,),
                )
                row = await cur.fetchone()
        if row is None:
            raise NotFoundError(f"Thread {thread_id} not found")
        return ThreadMetadata.model_validate(row["payload"])

    async def save_thread(
        self, thread: ThreadMetadata, context: RequestContext
    ) -> None:
        now = datetime.now(UTC)
        metadata = ThreadMetadata.model_validate(thread.model_dump(exclude={"items"}))
        payload = metadata.model_dump(mode="json")
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO chatkit_threads (id, title, created_at, updated_at, payload)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        updated_at = EXCLUDED.updated_at,
                        payload = EXCLUDED.payload
                    """,
                    (thread.id, thread.title, thread.created_at, now, Jsonb(payload)),
                )
            await conn.commit()

    async def load_threads(
        self,
        limit: int,
        after: str | None,
        order: str,
        context: RequestContext,
    ) -> Page[ThreadMetadata]:
        direction = _normalize_order(order)
        op = ">" if direction == "ASC" else "<"
        fetch_limit = max(1, limit) + 1

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                if after:
                    await cur.execute(
                        "SELECT created_at, id FROM chatkit_threads WHERE id = %s",
                        (after,),
                    )
                    cursor_row = await cur.fetchone()
                    if cursor_row is not None:
                        await cur.execute(
                            f"""
                            SELECT id, payload
                            FROM chatkit_threads
                            WHERE (created_at, id) {op} (%s, %s)
                            ORDER BY created_at {direction}, id {direction}
                            LIMIT %s
                            """,
                            (cursor_row["created_at"], cursor_row["id"], fetch_limit),
                        )
                    else:
                        await cur.execute(
                            f"""
                            SELECT id, payload
                            FROM chatkit_threads
                            ORDER BY created_at {direction}, id {direction}
                            LIMIT %s
                            """,
                            (fetch_limit,),
                        )
                else:
                    await cur.execute(
                        f"""
                        SELECT id, payload
                        FROM chatkit_threads
                        ORDER BY created_at {direction}, id {direction}
                        LIMIT %s
                        """,
                        (fetch_limit,),
                    )
                rows = await cur.fetchall()

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        data = [ThreadMetadata.model_validate(r["payload"]) for r in page_rows]
        next_after = page_rows[-1]["id"] if has_more and page_rows else None
        return Page(data=data, has_more=has_more, after=next_after)

    async def delete_thread(self, thread_id: str, context: RequestContext) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM chatkit_threads WHERE id = %s",
                    (thread_id,),
                )
            await conn.commit()

    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: RequestContext,
    ) -> Page[ThreadItem]:
        direction = _normalize_order(order)
        op = ">" if direction == "ASC" else "<"
        fetch_limit = max(1, limit) + 1

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                if after:
                    await cur.execute(
                        "SELECT created_at, seq FROM chatkit_items WHERE thread_id = %s AND id = %s",
                        (thread_id, after),
                    )
                    cursor_row = await cur.fetchone()
                    if cursor_row is not None:
                        await cur.execute(
                            f"""
                            SELECT id, payload
                            FROM chatkit_items
                            WHERE thread_id = %s AND (created_at, seq) {op} (%s, %s)
                            ORDER BY created_at {direction}, seq {direction}
                            LIMIT %s
                            """,
                            (
                                thread_id,
                                cursor_row["created_at"],
                                cursor_row["seq"],
                                fetch_limit,
                            ),
                        )
                    else:
                        await cur.execute(
                            f"""
                            SELECT id, payload
                            FROM chatkit_items
                            WHERE thread_id = %s
                            ORDER BY created_at {direction}, seq {direction}
                            LIMIT %s
                            """,
                            (thread_id, fetch_limit),
                        )
                else:
                    await cur.execute(
                        f"""
                        SELECT id, payload
                        FROM chatkit_items
                        WHERE thread_id = %s
                        ORDER BY created_at {direction}, seq {direction}
                        LIMIT %s
                        """,
                        (thread_id, fetch_limit),
                    )
                rows = await cur.fetchall()

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        data = [_THREAD_ITEM_ADAPTER.validate_python(r["payload"]) for r in page_rows]
        next_after = page_rows[-1]["id"] if has_more and page_rows else None
        return Page(data=data, has_more=has_more, after=next_after)

    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: RequestContext
    ) -> None:
        await self.save_item(thread_id, item, context)

    async def save_item(
        self, thread_id: str, item: ThreadItem, context: RequestContext
    ) -> None:
        now = datetime.now(UTC)
        payload = item.model_dump(mode="json")
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO chatkit_items (id, thread_id, created_at, payload)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        payload = EXCLUDED.payload
                    """,
                    (item.id, thread_id, item.created_at, Jsonb(payload)),
                )
                await cur.execute(
                    "UPDATE chatkit_threads SET updated_at = %s WHERE id = %s",
                    (now, thread_id),
                )
            await conn.commit()

    async def load_item(
        self, thread_id: str, item_id: str, context: RequestContext
    ) -> ThreadItem:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT payload FROM chatkit_items WHERE thread_id = %s AND id = %s",
                    (thread_id, item_id),
                )
                row = await cur.fetchone()
        if row is None:
            raise NotFoundError(f"Item {item_id} not found in thread {thread_id}")
        return _THREAD_ITEM_ADAPTER.validate_python(row["payload"])

    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: RequestContext
    ) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM chatkit_items WHERE thread_id = %s AND id = %s",
                    (thread_id, item_id),
                )
            await conn.commit()

    async def save_attachment(
        self, attachment: Attachment, context: RequestContext
    ) -> None:
        now = datetime.now(UTC)
        payload = attachment.model_dump(mode="json")
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO chatkit_attachments (id, created_at, payload)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        payload = EXCLUDED.payload
                    """,
                    (attachment.id, now, Jsonb(payload)),
                )
            await conn.commit()

    async def load_attachment(
        self, attachment_id: str, context: RequestContext
    ) -> Attachment:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT payload FROM chatkit_attachments WHERE id = %s",
                    (attachment_id,),
                )
                row = await cur.fetchone()
        if row is None:
            raise NotFoundError(f"Attachment {attachment_id} not found")
        return _ATTACHMENT_ADAPTER.validate_python(row["payload"])

    async def delete_attachment(
        self, attachment_id: str, context: RequestContext
    ) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM chatkit_attachments WHERE id = %s",
                    (attachment_id,),
                )
            await conn.commit()

    async def create_thread(
        self, title: str | None = None, context: RequestContext | None = None
    ) -> ThreadSummary:
        ctx = context or RequestContext()
        now = datetime.now(UTC)
        thread = ThreadMetadata(
            id=self.generate_thread_id(ctx),
            title=title,
            created_at=now,
        )
        await self.save_thread(thread, ctx)
        return ThreadSummary(
            id=thread.id,
            title=thread.title,
            created_at=now,
            updated_at=now,
            item_count=0,
            latest_phase=None,
        )

    async def list_thread_summaries(self, limit: int = 50) -> list[ThreadSummary]:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT
                        t.id,
                        t.title,
                        t.created_at,
                        t.updated_at,
                        COUNT(DISTINCT i.id) AS item_count,
                        (
                            SELECT r.phase
                            FROM research_runs r
                            WHERE r.thread_id = t.id
                            ORDER BY r.updated_at DESC
                            LIMIT 1
                        ) AS latest_phase
                    FROM chatkit_threads t
                    LEFT JOIN chatkit_items i ON i.thread_id = t.id
                    GROUP BY t.id
                    ORDER BY t.updated_at DESC, t.created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()

        return [
            ThreadSummary(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                item_count=int(row["item_count"]),
                latest_phase=ResearchPhase(row["latest_phase"])
                if row["latest_phase"]
                else None,
            )
            for row in rows
        ]

    async def upsert_research_run(
        self,
        *,
        thread_id: str,
        workflow_id: str,
        query: str,
        phase: ResearchPhase,
        message: str,
        sub_questions: list[str] | None = None,
        report: str | None = None,
        error: str | None = None,
    ) -> ResearchRunRecord:
        now = datetime.now(UTC)
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        questions = sub_questions or []

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    INSERT INTO research_runs (
                        id, thread_id, workflow_id, query, phase, message,
                        sub_questions, report, error, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (workflow_id) DO UPDATE SET
                        phase = EXCLUDED.phase,
                        message = EXCLUDED.message,
                        sub_questions = EXCLUDED.sub_questions,
                        report = COALESCE(EXCLUDED.report, research_runs.report),
                        error = EXCLUDED.error,
                        updated_at = EXCLUDED.updated_at
                    RETURNING *
                    """,
                    (
                        run_id,
                        thread_id,
                        workflow_id,
                        query,
                        phase.value,
                        message,
                        Jsonb(questions),
                        report,
                        error,
                        now,
                        now,
                    ),
                )
                row = await cur.fetchone()
            await conn.commit()

        assert row is not None
        return _row_to_run_record(row)

    async def list_research_runs(self, thread_id: str) -> list[ResearchRunRecord]:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT *
                    FROM research_runs
                    WHERE thread_id = %s
                    ORDER BY created_at DESC
                    """,
                    (thread_id,),
                )
                rows = await cur.fetchall()
        return [_row_to_run_record(row) for row in rows]

    async def list_active_research_runs(self) -> list[ResearchRunRecord]:
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT *
                    FROM research_runs
                    WHERE phase NOT IN (%s, %s)
                    ORDER BY created_at DESC
                    """,
                    (ResearchPhase.COMPLETED.value, ResearchPhase.FAILED.value),
                )
                rows = await cur.fetchall()
        return [_row_to_run_record(row) for row in rows]


def _row_to_run_record(row: dict[str, Any]) -> ResearchRunRecord:
    return ResearchRunRecord(
        id=row["id"],
        thread_id=row["thread_id"],
        workflow_id=row["workflow_id"],
        query=row["query"],
        phase=ResearchPhase(row["phase"]),
        message=row["message"],
        sub_questions=list(row["sub_questions"] or []),
        report=row["report"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
