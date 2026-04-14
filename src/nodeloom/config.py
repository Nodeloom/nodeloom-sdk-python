"""Configuration dataclass for the NodeLoom SDK."""

from dataclasses import dataclass, field
from typing import Optional


SDK_VERSION = "0.8.0"
SDK_LANGUAGE = "python"

DEFAULT_ENDPOINT = "https://api.nodeloom.io"
DEFAULT_BATCH_SIZE = 100
DEFAULT_FLUSH_INTERVAL = 5.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_QUEUE_MAX_SIZE = 10000
DEFAULT_TIMEOUT = 10.0


@dataclass(frozen=True)
class NodeLoomConfig:
    """Immutable configuration for the NodeLoom client.

    Attributes:
        api_key: API key for authentication (must start with "sdk_").
        endpoint: Base URL of the NodeLoom telemetry API.
        environment: Deployment environment label (e.g. "production", "staging").
        batch_size: Maximum number of events per batch request.
        flush_interval: Maximum seconds between batch flushes.
        max_retries: Number of retry attempts for failed requests.
        queue_max_size: Upper bound on the in-memory event queue.
        timeout: HTTP request timeout in seconds.
        enabled: Global kill switch. When False, no events are sent.
    """

    api_key: str
    endpoint: str = DEFAULT_ENDPOINT
    environment: str = "production"
    batch_size: int = DEFAULT_BATCH_SIZE
    flush_interval: float = DEFAULT_FLUSH_INTERVAL
    max_retries: int = DEFAULT_MAX_RETRIES
    queue_max_size: int = DEFAULT_QUEUE_MAX_SIZE
    timeout: float = DEFAULT_TIMEOUT
    enabled: bool = True

    def __repr__(self) -> str:
        masked_key = self.api_key[:6] + "***" if self.api_key and len(self.api_key) > 6 else "***"
        return (f"NodeLoomConfig(api_key='{masked_key}', endpoint='{self.endpoint}', "
                f"environment='{self.environment}')")

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api_key is required")
        if self.endpoint and not self.endpoint.startswith("https://") and "localhost" not in self.endpoint and "127.0.0.1" not in self.endpoint:
            import warnings
            warnings.warn(
                f"NodeLoom endpoint '{self.endpoint}' does not use HTTPS. "
                "API keys will be sent in plaintext. Use HTTPS in production.",
                stacklevel=2,
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.flush_interval <= 0:
            raise ValueError("flush_interval must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.queue_max_size < 1:
            raise ValueError("queue_max_size must be at least 1")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
