"""HTTP transport layer with retry and exponential backoff."""

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from nodeloom.config import NodeLoomConfig, SDK_VERSION, SDK_LANGUAGE

logger = logging.getLogger("nodeloom.transport")


class HttpTransport:
    """Sends batched telemetry events to the NodeLoom API.

    Uses exponential backoff with jitter for retries. All failures are
    logged but never raised, keeping the SDK fire-and-forget.
    """

    TELEMETRY_PATH = "/api/sdk/v1/telemetry"

    def __init__(self, config: NodeLoomConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
                "User-Agent": f"nodeloom-python-sdk/{SDK_VERSION}",
            }
        )

    @property
    def url(self) -> str:
        base = self._config.endpoint.rstrip("/")
        return f"{base}{self.TELEMETRY_PATH}"

    def send_batch(self, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Send a batch of events.

        Returns the parsed response body on success, or None if all
        attempts fail. Exceptions are caught and logged internally.
        """
        if not events:
            return None

        payload = {
            "events": events,
            "sdk_version": SDK_VERSION,
            "sdk_language": SDK_LANGUAGE,
        }

        last_error: Optional[Exception] = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._session.post(
                    self.url,
                    json=payload,
                    timeout=self._config.timeout,
                )
                if response.status_code == 200:
                    body = response.json()
                    rejected = body.get("rejected", 0)
                    if rejected > 0:
                        logger.warning(
                            "Batch partially rejected: %d rejected out of %d",
                            rejected,
                            len(events),
                        )
                        for err in body.get("errors", []):
                            logger.warning(
                                "  event[%s]: %s",
                                err.get("index"),
                                err.get("error"),
                            )
                    return body

                # Retry on server errors (5xx) and 429 (rate limit).
                # Other client errors (4xx) are not retryable.
                if 500 <= response.status_code < 600 or response.status_code == 429:
                    last_error = Exception(
                        f"Server error {response.status_code}: {response.text}"
                    )
                    logger.warning(
                        "Batch send failed (attempt %d/%d): HTTP %d",
                        attempt + 1,
                        self._config.max_retries + 1,
                        response.status_code,
                    )
                else:
                    # Client errors are not retryable
                    logger.error(
                        "Batch send failed with client error HTTP %d: %s",
                        response.status_code,
                        response.text,
                    )
                    return None

            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Batch send failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._config.max_retries + 1,
                    exc,
                )

            # Exponential backoff: 1s, 2s, 4s ...
            if attempt < self._config.max_retries:
                backoff = 2**attempt
                time.sleep(backoff)

        logger.error(
            "Batch send failed after %d attempts. Last error: %s",
            self._config.max_retries + 1,
            last_error,
        )
        return None

    def close(self) -> None:
        """Close the underlying HTTP session."""
        try:
            self._session.close()
        except Exception:
            pass
