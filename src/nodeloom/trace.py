"""Trace represents a top-level execution of an AI agent."""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING

from nodeloom.control import ControlRegistry, raise_if_halted
from nodeloom.span import Span
from nodeloom.types import SpanType, TraceStatus

if TYPE_CHECKING:
    from nodeloom.queue import TelemetryQueue

logger = logging.getLogger("nodeloom.trace")


class Trace:
    """A trace groups multiple spans under a single agent execution.

    Traces are NOT thread-safe. Each trace should be driven from one
    thread. Use as a context manager for automatic lifecycle management:

        with client.trace("my-agent", input={"query": "hi"}) as t:
            with t.span("llm-call", type=SpanType.LLM) as s:
                s.set_output({"response": "hello"})
    """

    def __init__(
        self,
        agent_name: str,
        queue: "TelemetryQueue",
        input_data: Optional[Dict[str, Any]] = None,
        agent_version: Optional[str] = None,
        environment: str = "production",
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        framework: Optional[str] = None,
        framework_version: Optional[str] = None,
        control_registry: Optional[ControlRegistry] = None,
    ) -> None:
        self._control_registry = control_registry
        # Fail-fast on halted agents BEFORE allocating any state, so callers
        # see AgentHaltedError as a synchronous, no-side-effect failure.
        if control_registry is not None:
            raise_if_halted(control_registry, agent_name)

        self._trace_id = str(uuid.uuid4())
        self._agent_name = agent_name
        self._agent_version = agent_version
        self._environment = environment
        self._session_id = session_id
        self._framework = framework
        self._framework_version = framework_version
        self._queue = queue

        self._input = input_data
        self._output: Optional[Dict[str, Any]] = None
        self._metadata = metadata
        self._error: Optional[str] = None
        self._status: Optional[TraceStatus] = None

        self._timestamp = datetime.now(timezone.utc).isoformat()
        self._ended = False

        # Emit trace_start event immediately
        self._emit_start()

    # -- Properties ----------------------------------------------------------

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def ended(self) -> bool:
        return self._ended

    # -- Span factory --------------------------------------------------------

    def span(
        self,
        name: str,
        type: SpanType = SpanType.CUSTOM,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        """Create a new span within this trace.

        Args:
            name: Human-readable label for the operation.
            type: Classification of the span (LLM, TOOL, etc.).
            parent_span_id: Optional ID of a parent span for nesting.

        Returns:
            A new Span instance (also usable as a context manager).
        """
        if self._ended:
            logger.warning(
                "Creating span on an already-ended trace %s", self._trace_id
            )
        return Span(
            name=name,
            trace_id=self._trace_id,
            queue=self._queue,
            span_type=type,
            parent_span_id=parent_span_id,
        )

    # -- Events --------------------------------------------------------------

    def event(
        self,
        name: str,
        level: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a standalone event attached to this trace."""
        evt: Dict[str, Any] = {
            "type": "event",
            "trace_id": self._trace_id,
            "name": name,
            "level": level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if data is not None:
            evt["data"] = data
        self._queue.put(evt)

    # -- Lifecycle -----------------------------------------------------------

    def end(
        self,
        status: TraceStatus = TraceStatus.SUCCESS,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Finalize the trace and enqueue a trace_end event.

        Calling ``end()`` more than once is a no-op (with a warning).
        """
        if self._ended:
            logger.warning("Trace %s already ended", self._trace_id)
            return

        self._ended = True
        self._status = status
        if output is not None:
            self._output = output
        if error is not None:
            self._error = error

        event: Dict[str, Any] = {
            "type": "trace_end",
            "trace_id": self._trace_id,
            "status": status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self._output is not None:
            event["output"] = self._output
        if self._error is not None:
            event["error"] = self._error

        self._queue.put(event)

    def _emit_start(self) -> None:
        """Enqueue a trace_start event."""
        event: Dict[str, Any] = {
            "type": "trace_start",
            "trace_id": self._trace_id,
            "agent_name": self._agent_name,
            "environment": self._environment,
            "timestamp": self._timestamp,
        }
        if self._agent_version is not None:
            event["agent_version"] = self._agent_version
        if self._input is not None:
            event["input"] = self._input
        if self._metadata is not None:
            event["metadata"] = self._metadata
        if self._session_id is not None:
            event["session_id"] = self._session_id
        if self._framework is not None:
            event["framework"] = self._framework
        if self._framework_version is not None:
            event["framework_version"] = self._framework_version
        event["sdk_language"] = "python"

        # Phase 2: attach the cached guardrail session id so HARD-mode
        # required-guardrail enforcement on the backend can correlate this
        # trace with a recent check_guardrails call.
        if self._control_registry is not None:
            session_id = self._control_registry.take_guardrail_session(
                self._agent_name, time.monotonic()
            )
            if session_id:
                event["guardrail_session_id"] = session_id

        self._queue.put(event)

    # -- Feedback ------------------------------------------------------------

    def feedback(self, rating: int, comment: Optional[str] = None) -> None:
        """Submit feedback for this trace.

        Args:
            rating: Rating from 1 to 5.
            comment: Optional comment.
        """
        evt: Dict[str, Any] = {
            "type": "feedback",
            "trace_id": self._trace_id,
            "rating": rating,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if comment is not None:
            evt["comment"] = comment
        self._queue.put(evt)

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if not self._ended:
            if exc_type is not None:
                self.end(
                    status=TraceStatus.ERROR,
                    error=f"{exc_type.__name__}: {exc_val}",
                )
            else:
                self.end(status=TraceStatus.SUCCESS)
        return None
