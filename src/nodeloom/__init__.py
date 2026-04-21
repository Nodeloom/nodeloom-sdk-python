"""NodeLoom Python SDK for AI agent monitoring and telemetry."""

from nodeloom.api import ApiClient, ApiError
from nodeloom.client import NodeLoomClient
from nodeloom.config import SDK_VERSION
from nodeloom.control import AgentControlState, AgentHaltedError, ControlRegistry
from nodeloom.span import Span
from nodeloom.trace import Trace
from nodeloom.types import EventLevel, SpanType, TraceStatus

# Convenience alias
NodeLoom = NodeLoomClient

__version__ = SDK_VERSION

__all__ = [
    "AgentControlState",
    "AgentHaltedError",
    "ApiClient",
    "ApiError",
    "ControlRegistry",
    "NodeLoom",
    "NodeLoomClient",
    "Trace",
    "Span",
    "SpanType",
    "TraceStatus",
    "EventLevel",
    "__version__",
]
