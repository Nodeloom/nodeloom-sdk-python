"""Optional standalone poll thread that refreshes the control registry.

Telemetry batch responses already piggy-back the control payload, so this
poller is mainly useful for sparse-traffic agents that may go minutes
between traces. It periodically calls ``GET /api/sdk/v1/agents/{name}/control``
for every agent that has emitted at least one trace_start.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

from nodeloom.control import ControlRegistry

if TYPE_CHECKING:
    from nodeloom.api import ApiClient

logger = logging.getLogger("nodeloom.control_poller")


class ControlPoller:
    """Background thread that refreshes :class:`ControlRegistry` periodically."""

    def __init__(self, registry: ControlRegistry, api_factory, interval_seconds: float) -> None:
        # ``api_factory`` is a zero-arg callable that returns the ApiClient. We
        # take a factory rather than the client itself so the client can be
        # lazily constructed (the existing NodeLoomClient lazies it too).
        self._registry = registry
        self._api_factory = api_factory
        self._interval = max(1.0, float(interval_seconds))
        self._shutdown = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="nodeloom-control-poller",
            daemon=True,
        )
        self._thread.start()
        logger.debug("ControlPoller started (interval=%.1fs)", self._interval)

    def shutdown(self, timeout: float = 2.0) -> None:
        self._shutdown.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._shutdown.wait(timeout=self._interval):
            try:
                api = self._api_factory()
            except Exception:  # pragma: no cover - defensive
                logger.exception("ControlPoller failed to build API client")
                continue

            for agent_name in self._registry.known_agents():
                if self._shutdown.is_set():
                    break
                try:
                    api.get_agent_control(agent_name)
                except Exception as exc:
                    # Polling is best-effort. Failures are logged once and not
                    # retried — the next tick will try again.
                    logger.debug("Control poll failed for %s: %s", agent_name, exc)
