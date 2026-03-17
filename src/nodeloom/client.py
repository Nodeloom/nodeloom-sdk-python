"""NodeLoomClient is the main entry point for the SDK."""

import logging
from typing import Any, Dict, Optional

from nodeloom.batch_processor import BatchProcessor
from nodeloom.config import NodeLoomConfig
from nodeloom.queue import TelemetryQueue
from nodeloom.trace import Trace
from nodeloom.transport import HttpTransport
from nodeloom.types import EventLevel

logger = logging.getLogger("nodeloom")


class NodeLoomClient:
    """Thread-safe client for sending telemetry to NodeLoom.

    Usage::

        from nodeloom import NodeLoom

        client = NodeLoom(api_key="sdk_abc123")
        with client.trace("my-agent", input={"query": "hello"}) as t:
            with t.span("llm-call", type=SpanType.LLM) as s:
                s.set_output({"text": "world"})
                s.set_token_usage(prompt=10, completion=20, model="gpt-4o")
        client.shutdown()

    The client is thread-safe: multiple threads may create traces
    concurrently. Individual Trace and Span objects are not thread-safe
    and should be used from a single thread.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.nodeloom.io",
        environment: str = "production",
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_retries: int = 3,
        queue_max_size: int = 10000,
        timeout: float = 10.0,
        enabled: bool = True,
    ) -> None:
        self._config = NodeLoomConfig(
            api_key=api_key,
            endpoint=endpoint,
            environment=environment,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_retries=max_retries,
            queue_max_size=queue_max_size,
            timeout=timeout,
            enabled=enabled,
        )
        self._queue = TelemetryQueue(max_size=queue_max_size)
        self._transport = HttpTransport(self._config)
        self._processor = BatchProcessor(
            config=self._config,
            telemetry_queue=self._queue,
            transport=self._transport,
        )
        self._shutdown_called = False

        self._api: Optional["ApiClient"] = None

        if enabled:
            self._processor.start()

    # -- Properties ----------------------------------------------------------

    @property
    def api(self) -> "ApiClient":
        """Access the REST API client.

        Uses the same API key and endpoint as the telemetry client.
        SDK tokens can now authenticate against all NodeLoom API endpoints.
        """
        if self._api is None:
            from nodeloom.api import ApiClient
            self._api = ApiClient(
                api_key=self._config.api_key,
                endpoint=self._config.endpoint,
            )
        return self._api

    @property
    def config(self) -> NodeLoomConfig:
        return self._config

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    # -- Public API ----------------------------------------------------------

    def trace(
        self,
        agent_name: str,
        input: Optional[Dict[str, Any]] = None,
        agent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Trace:
        """Start a new trace for an agent execution.

        Args:
            agent_name: Name of the agent being traced.
            input: The input data for this execution.
            agent_version: Optional version identifier for the agent.
            metadata: Arbitrary key-value metadata to attach.

        Returns:
            A Trace object (also usable as a context manager).
        """
        return Trace(
            agent_name=agent_name,
            queue=self._queue,
            input_data=input,
            agent_version=agent_version,
            environment=self._config.environment,
            metadata=metadata,
        )

    def event(
        self,
        name: str,
        level: EventLevel = EventLevel.INFO,
        data: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """Emit a standalone event, optionally associated with a trace.

        Args:
            name: Event name (e.g. "guardrail_triggered").
            level: Severity level.
            data: Arbitrary event payload.
            trace_id: Optional trace to attach this event to.
        """
        from datetime import datetime, timezone

        evt: Dict[str, Any] = {
            "type": "event",
            "trace_id": trace_id,
            "name": name,
            "level": level.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if data is not None:
            evt["data"] = data
        self._queue.put(evt)

    def flush(self) -> None:
        """Request an immediate flush of all queued events.

        This is non-blocking. It signals the background processor to
        wake up and send pending events.
        """
        if self._config.enabled:
            self._processor.flush()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Flush remaining events and stop the background processor.

        Args:
            timeout: Maximum seconds to wait for the final flush.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if self._api is not None:
            self._api.close()
        logger.debug("NodeLoomClient shutting down")

        if self._config.enabled:
            self._processor.shutdown(timeout=timeout)
        self._transport.close()
