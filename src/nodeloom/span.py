"""Span represents a single operation within a trace."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING

from nodeloom.types import SpanType, TraceStatus

if TYPE_CHECKING:
    from nodeloom.queue import TelemetryQueue

logger = logging.getLogger("nodeloom.span")


class Span:
    """A single timed operation within a trace.

    Spans are NOT thread-safe. Each span should be used from a single
    thread. Use as a context manager for automatic ``end()`` calls:

        with trace.span("my-operation", type=SpanType.LLM) as s:
            s.set_input({"prompt": "Hello"})
            result = call_llm()
            s.set_output(result)
    """

    def __init__(
        self,
        name: str,
        trace_id: str,
        queue: "TelemetryQueue",
        span_type: SpanType = SpanType.CUSTOM,
        parent_span_id: Optional[str] = None,
    ) -> None:
        self._span_id = str(uuid.uuid4())
        self._trace_id = trace_id
        self._name = name
        self._span_type = span_type
        self._parent_span_id = parent_span_id
        self._queue = queue

        self._input: Optional[Dict[str, Any]] = None
        self._output: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._status: TraceStatus = TraceStatus.SUCCESS
        self._token_usage: Optional[Dict[str, Any]] = None
        self._prompt_template: Optional[str] = None
        self._prompt_version: Optional[int] = None

        self._timestamp = datetime.now(timezone.utc).isoformat()
        self._end_timestamp: Optional[str] = None
        self._ended = False

    # -- Properties ----------------------------------------------------------

    @property
    def span_id(self) -> str:
        return self._span_id

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def parent_span_id(self) -> Optional[str]:
        return self._parent_span_id

    @property
    def ended(self) -> bool:
        return self._ended

    # -- Setters -------------------------------------------------------------

    def set_input(self, input_data: Dict[str, Any]) -> "Span":
        """Attach input data to this span. Returns self for chaining."""
        self._input = input_data
        return self

    def set_output(self, output_data: Dict[str, Any]) -> "Span":
        """Attach output data to this span. Returns self for chaining."""
        self._output = output_data
        return self

    def set_error(self, error: str) -> "Span":
        """Record an error message and mark the span as errored."""
        self._error = error
        self._status = TraceStatus.ERROR
        return self

    def set_token_usage(
        self,
        prompt: int = 0,
        completion: int = 0,
        model: Optional[str] = None,
    ) -> "Span":
        """Record token consumption for LLM spans."""
        self._token_usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
        if model:
            self._token_usage["model"] = model
        return self

    def set_prompt(self, template: str, version: int) -> "Span":
        """Record which prompt template and version was used."""
        self._prompt_template = template
        self._prompt_version = version
        return self

    def metric(
        self,
        name: str,
        value: float,
        unit: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> "Span":
        """Emit a custom metric tied to this span's trace."""
        evt: Dict[str, Any] = {
            "type": "metric",
            "trace_id": self._trace_id,
            "metric_name": name,
            "metric_value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if unit is not None:
            evt["metric_unit"] = unit
        if tags is not None:
            evt["metric_tags"] = tags
        self._queue.put(evt)
        return self

    # -- Lifecycle -----------------------------------------------------------

    def end(
        self,
        status: Optional[TraceStatus] = None,
        output: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finalize the span and enqueue it for sending.

        Calling ``end()`` more than once is a no-op (with a warning).
        """
        if self._ended:
            logger.warning("Span %s (%s) already ended", self._name, self._span_id)
            return

        self._ended = True
        self._end_timestamp = datetime.now(timezone.utc).isoformat()

        if status is not None:
            self._status = status
        if output is not None:
            self._output = output

        event = self._build_event()
        self._queue.put(event)

    def _build_event(self) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "type": "span",
            "trace_id": self._trace_id,
            "span_id": self._span_id,
            "parent_span_id": self._parent_span_id,
            "name": self._name,
            "span_type": self._span_type.value,
            "status": self._status.value,
            "timestamp": self._timestamp,
            "end_timestamp": self._end_timestamp,
        }
        if self._input is not None:
            event["input"] = self._input
        if self._output is not None:
            event["output"] = self._output
        if self._error is not None:
            event["error"] = self._error
        if self._token_usage is not None:
            event["token_usage"] = self._token_usage
        if self._prompt_template is not None:
            event["prompt_template"] = self._prompt_template
        if self._prompt_version is not None:
            event["prompt_version"] = self._prompt_version
        return event

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if not self._ended:
            if exc_type is not None:
                self.set_error(f"{exc_type.__name__}: {exc_val}")
                self.end(status=TraceStatus.ERROR)
            else:
                self.end()
        # Do not suppress exceptions
        return None
