"""Background daemon thread that batches and sends telemetry events."""

import logging
import threading
from typing import Optional

from nodeloom.config import NodeLoomConfig
from nodeloom.queue import TelemetryQueue
from nodeloom.transport import HttpTransport

logger = logging.getLogger("nodeloom.batch_processor")


class BatchProcessor:
    """Consumes events from a TelemetryQueue, batches them, and sends
    them via HttpTransport.

    Batches are flushed when either of the following conditions is met:
      - The batch reaches ``config.batch_size`` events.
      - ``config.flush_interval`` seconds have elapsed since the last flush.

    The processor runs on a daemon thread, so it will not prevent the
    interpreter from exiting. Call ``shutdown()`` to flush remaining
    events before exit.
    """

    def __init__(
        self,
        config: NodeLoomConfig,
        telemetry_queue: TelemetryQueue,
        transport: HttpTransport,
    ) -> None:
        self._config = config
        self._queue = telemetry_queue
        self._transport = transport
        self._shutdown_event = threading.Event()
        self._flush_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background processing thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="nodeloom-batch-processor",
            daemon=True,
        )
        self._thread.start()
        logger.debug("BatchProcessor started")

    def _run(self) -> None:
        """Main loop for the background thread."""
        while not self._shutdown_event.is_set():
            # Wait for either the flush interval to elapse or a manual
            # flush signal, whichever comes first.
            self._flush_event.wait(timeout=self._config.flush_interval)
            self._flush_event.clear()
            self._flush_batch()

        # Final flush on shutdown
        self._flush_batch()
        logger.debug("BatchProcessor stopped")

    def _flush_batch(self) -> None:
        """Drain events from the queue and send them in batches."""
        while not self._queue.is_empty():
            events = self._queue.drain(self._config.batch_size)
            if not events:
                break
            try:
                self._transport.send_batch(events)
            except Exception:
                logger.exception("Unexpected error while sending batch")

    def flush(self) -> None:
        """Request an immediate flush of pending events.

        This signals the background thread to wake up and process
        whatever is currently in the queue. It does not block until
        the flush completes.
        """
        self._flush_event.set()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Signal the processor to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for the background thread
                     to complete its final flush.
        """
        logger.debug("BatchProcessor shutting down")
        self._shutdown_event.set()
        self._flush_event.set()  # Wake the thread if it is sleeping
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "BatchProcessor thread did not terminate within %.1f seconds",
                    timeout,
                )
