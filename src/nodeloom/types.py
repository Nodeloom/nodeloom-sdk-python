"""Enumerations and type definitions for the NodeLoom SDK."""

from enum import Enum


class SpanType(str, Enum):
    """Classification of span operations."""

    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    CHAIN = "chain"
    AGENT = "agent"
    CUSTOM = "custom"


class TraceStatus(str, Enum):
    """Terminal status of a trace or span."""

    SUCCESS = "success"
    ERROR = "error"


class EventLevel(str, Enum):
    """Severity level for standalone events."""

    INFO = "info"
    WARN = "warn"
    ERROR = "error"
