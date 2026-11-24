from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.config import Settings
from app.events import InMemoryThreadEventBus
from app.main import create_app
from app.temporal.client import TemporalClientManager, TemporalUnavailableError
from tests.test_api import InMemoryChatKitStore


def _make_fake_client(*, healthy: bool = True) -> MagicMock:
    client = MagicMock()
    client.service_client.check_health = AsyncMock(return_value=healthy)
    return client


@pytest.mark.asyncio
async def test_get_client_connects_once_under_concurrency() -> None:
    fake_client = _make_fake_client()
    connect_calls = 0

    async def fake_connect(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal connect_calls
        connect_calls += 1
        await asyncio.sleep(0.02)
        return fake_client

    manager = TemporalClientManager(Settings(temporal_address="localhost:7233"))
    with patch("app.temporal.client.Client.connect", side_effect=fake_connect):
        results = await asyncio.gather(*(manager.get_client() for _ in range(5)))

    assert connect_calls == 1
    assert all(result is fake_client for result in results)
    assert manager.is_connected is True


@pytest.mark.asyncio
async def test_get_client_raises_unavailable_and_retries_later() -> None:
    fake_client = _make_fake_client()
    attempts = 0

    async def flaky_connect(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionRefusedError("Temporal not ready")
        return fake_client

    manager = TemporalClientManager(Settings(temporal_address="localhost:7233"))
    with patch("app.temporal.client.Client.connect", side_effect=flaky_connect):
        with pytest.raises(TemporalUnavailableError):
            await manager.get_client()
        assert manager.is_connected is False

        client = await manager.get_client()
        assert client is fake_client
        assert manager.is_connected is True
        assert attempts == 2


@pytest.mark.asyncio
async def test_ping_checks_real_health_and_close_clears_cache() -> None:
    fake_client = _make_fake_client(healthy=True)
    manager = TemporalClientManager.from_client(fake_client)

    assert await manager.ping() is True
    fake_client.service_client.check_health.assert_awaited_once()

    fake_client.service_client.check_health.return_value = False
    assert await manager.ping() is False

    await manager.close()
    assert manager.is_connected is False
    with pytest.raises(TemporalUnavailableError):
        await manager.get_client()


@pytest.mark.asyncio
async def test_api_handles_temporal_unavailable_gracefully() -> None:
    store = InMemoryChatKitStore()
    bus = InMemoryThreadEventBus()
    manager = TemporalClientManager(Settings(temporal_address="127.0.0.1:1"))

    with patch(
        "app.temporal.client.Client.connect",
        side_effect=ConnectionRefusedError("Temporal offline"),
    ):
        app = create_app(
            settings=Settings(openai_api_key="sk-test"),
            store=store,
            event_bus=bus,
            temporal_client_manager=manager,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                health_res = await client.get("/health")
                assert health_res.status_code == 200
                assert health_res.json()["temporal"] is False
                assert health_res.json()["postgres"] is True

                state_res = await client.get("/api/threads/thr_missing/state")
                assert state_res.status_code == 503

                chat_res = await client.post(
                    "/chatkit",
                    json={
                        "type": "threads.create",
                        "params": {
                            "input": {
                                "content": [
                                    {"type": "input_text", "text": "Hello Temporal"}
                                ],
                                "attachments": [],
                                "inference_options": {},
                            }
                        },
                    },
                )
                assert chat_res.status_code == 200
                events = [
                    json.loads(line[len("data: ") :])
                    for line in chat_res.text.splitlines()
                    if line.startswith("data: ")
                ]
                assert any(
                    e["type"] == "thread.item.done"
                    and e["item"]["type"] == "assistant_message"
                    and "currently unavailable" in e["item"]["content"][0]["text"].lower()
                    for e in events
                )
