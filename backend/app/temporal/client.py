from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any

from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin

from app.config import Settings

logger = logging.getLogger(__name__)


class TemporalUnavailableError(RuntimeError):
    """Raised when a Temporal client connection cannot be established."""


def build_openai_agents_plugin(
    *, model_provider: Any | None = None
) -> OpenAIAgentsPlugin:
    return OpenAIAgentsPlugin(
        model_params=ModelActivityParameters(
            start_to_close_timeout=timedelta(seconds=180),
            schedule_to_close_timeout=timedelta(seconds=420),
            retry_policy=RetryPolicy(
                backoff_coefficient=2.0,
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=3,
            ),
        ),
        model_provider=model_provider,
    )


class TemporalClientManager:
    """Single owner of Temporal client creation, plugin configuration, caching, and health checks."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_provider: Any | None = None,
        client: Client | None = None,
    ) -> None:
        self._settings = settings
        self._model_provider = model_provider
        self._client: Client | None = client
        self._lock = asyncio.Lock()

    @classmethod
    def from_client(
        cls,
        client: Client,
        settings: Settings | None = None,
    ) -> TemporalClientManager:
        return cls(settings=settings, client=client)

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    async def get_client(self) -> Client:
        """Return the cached Temporal client, or lazily connect once under a lock."""
        if self._client is not None:
            return self._client
        async with self._lock:
            if self._client is not None:
                return self._client
            if self._settings is None:
                raise TemporalUnavailableError(
                    "TemporalClientManager has neither settings nor an injected client."
                )
            try:
                self._client = await Client.connect(
                    self._settings.temporal_address,
                    namespace=self._settings.temporal_namespace,
                    plugins=[
                        build_openai_agents_plugin(model_provider=self._model_provider)
                    ],
                )
            except Exception as exc:
                raise TemporalUnavailableError(
                    f"Unable to connect to Temporal at {self._settings.temporal_address}: {exc}"
                ) from exc
            return self._client

    async def ping(self, *, timeout: timedelta = timedelta(seconds=3)) -> bool:
        """Ensure connected and run a real gRPC health check against Temporal."""
        try:
            client = await self.get_client()
            return await client.service_client.check_health(timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Temporal health check failed: %s", exc)
            return False

    async def close(self) -> None:
        """Clear the cached client reference on shutdown."""
        async with self._lock:
            self._client = None
