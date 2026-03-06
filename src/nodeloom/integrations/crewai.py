"""CrewAI integration for NodeLoom telemetry.

Provides decorator-based hooks that create traces for crew runs and
spans for individual agent tasks.

Usage::

    from nodeloom import NodeLoom, SpanType
    from nodeloom.integrations.crewai import instrument_crew

    client = NodeLoom(api_key="sdk_...")

    @instrument_crew(client, agent_name="my-crew")
    def run_crew():
        crew = Crew(agents=[...], tasks=[...])
        return crew.kickoff()

    # Or use the CrewAIInstrumentation class directly:
    instrumentation = CrewAIInstrumentation(client)
    with instrumentation.trace_crew("my-crew", input={"query": "..."}) as crew_trace:
        with crew_trace.task("research", agent="researcher") as task_span:
            result = do_research()
            task_span.set_output({"result": result})

Requires ``crewai`` to be installed. If it is not available, the
``instrument_crew`` decorator will still work (it wraps function calls
with traces), but the deeper hooks will not activate.
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from nodeloom.client import NodeLoomClient
from nodeloom.span import Span
from nodeloom.trace import Trace
from nodeloom.types import SpanType, TraceStatus

logger = logging.getLogger("nodeloom.integrations.crewai")

F = TypeVar("F", bound=Callable[..., Any])


class CrewAIInstrumentation:
    """Manual instrumentation helper for CrewAI workflows.

    Provides context managers that map crew runs to traces and
    individual tasks to spans.

    Example::

        inst = CrewAIInstrumentation(client)
        with inst.trace_crew("my-crew") as crew_ctx:
            with crew_ctx.task("research", agent="researcher") as span:
                result = do_research()
                span.set_output({"result": result})
            with crew_ctx.task("write", agent="writer") as span:
                article = do_writing()
                span.set_output({"article": article})
    """

    def __init__(self, client: NodeLoomClient) -> None:
        self._client = client

    def trace_crew(
        self,
        crew_name: str,
        input: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CrewTraceContext":
        """Start a traced crew execution.

        Returns a CrewTraceContext that can be used as a context manager
        and also provides a ``task()`` method for creating task spans.
        """
        trace = self._client.trace(
            agent_name=crew_name,
            input=input,
            metadata=metadata,
        )
        return CrewTraceContext(trace)


class CrewTraceContext:
    """Context manager wrapping a trace for a CrewAI crew execution."""

    def __init__(self, trace: Trace) -> None:
        self._trace = trace

    @property
    def trace(self) -> Trace:
        return self._trace

    def task(
        self,
        task_name: str,
        agent: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        """Create a span for a CrewAI task.

        Args:
            task_name: Name of the task being executed.
            agent: Name of the agent performing the task (stored as input).
            parent_span_id: Optional parent span for nesting.

        Returns:
            A Span that can be used as a context manager.
        """
        span = self._trace.span(
            name=task_name,
            type=SpanType.AGENT,
            parent_span_id=parent_span_id,
        )
        if agent:
            span.set_input({"agent": agent})
        return span

    def __enter__(self) -> "CrewTraceContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if not self._trace.ended:
            if exc_type is not None:
                self._trace.end(
                    status=TraceStatus.ERROR,
                    error=f"{exc_type.__name__}: {exc_val}",
                )
            else:
                self._trace.end(status=TraceStatus.SUCCESS)
        return None


def instrument_crew(
    client: NodeLoomClient,
    agent_name: str = "crewai-agent",
    agent_version: Optional[str] = None,
) -> Callable[[F], F]:
    """Decorator that wraps a function with a NodeLoom trace.

    The decorated function's return value is captured as the trace
    output. If the function raises an exception, the trace is marked
    as errored. The exception is always re-raised.

    Example::

        @instrument_crew(client, agent_name="my-crew")
        def run_my_crew(query: str):
            crew = Crew(agents=[...], tasks=[...])
            return crew.kickoff(inputs={"query": query})

        result = run_my_crew("What is the weather?")
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build input from function arguments
            input_data: Dict[str, Any] = {}
            if args:
                input_data["args"] = [str(a) for a in args]
            if kwargs:
                input_data["kwargs"] = {k: str(v) for k, v in kwargs.items()}

            with client.trace(
                agent_name=agent_name,
                input=input_data if input_data else None,
                agent_version=agent_version,
            ) as trace:
                with trace.span(
                    name=func.__name__, type=SpanType.AGENT
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        output = (
                            result
                            if isinstance(result, dict)
                            else {"result": str(result)}
                        )
                        span.set_output(output)
                        span.end(status=TraceStatus.SUCCESS, output=output)
                        return result
                    except Exception as exc:
                        span.set_error(str(exc))
                        span.end(status=TraceStatus.ERROR)
                        raise

        return cast(F, wrapper)

    return decorator
