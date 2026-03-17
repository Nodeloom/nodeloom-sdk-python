"""Configuration dataclass for the NodeLoom SDK."""

from dataclasses import dataclass, field
from typing import Optional


SDK_VERSION = "0.2.0"
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

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api_key is required")
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
