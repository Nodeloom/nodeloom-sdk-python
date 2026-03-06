"""Thread-safe bounded event queue for the NodeLoom SDK."""

import logging
import queue
from typing import Any, Dict, List

logger = logging.getLogger("nodeloom.queue")


class TelemetryQueue:
    """A bounded, thread-safe wrapper around queue.Queue.

    Events that arrive when the queue is full are silently dropped
    with a warning log. This prevents unbounded memory growth if the
    consumer (BatchProcessor) cannot keep up.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max_size)
        self._max_size = max_size
        self._dropped_count = 0

    @property
    def dropped_count(self) -> int:
        """Number of events dropped due to a full queue."""
        return self._dropped_count

    def put(self, event: Dict[str, Any]) -> bool:
        """Enqueue an event. Returns True on success, False if the queue is full."""
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            self._dropped_count += 1
            if self._dropped_count % 100 == 1:
                logger.warning(
                    "Telemetry queue is full (max %d). Event dropped. "
                    "Total dropped so far: %d",
                    self._max_size,
                    self._dropped_count,
                )
            return False

    def drain(self, max_items: int) -> List[Dict[str, Any]]:
        """Remove and return up to max_items events from the queue.

        This is non-blocking. It returns whatever is available immediately
        (up to max_items).
        """
        items: List[Dict[str, Any]] = []
        for _ in range(max_items):
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def size(self) -> int:
        """Approximate number of events currently in the queue."""
        return self._queue.qsize()

    def is_empty(self) -> bool:
        """Check whether the queue is empty."""
        return self._queue.empty()
