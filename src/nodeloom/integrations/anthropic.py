"""Anthropic Managed Agents integration for NodeLoom.

Auto-instruments Anthropic Managed Agent sessions with traces, spans,
guardrail checks, and token tracking.

Usage:
    from nodeloom import NodeLoom
    from nodeloom.integrations.anthropic import ManagedAgentsHandler
    from anthropic import Anthropic

    nodeloom = NodeLoom(api_key="sdk_...")
    anthropic_client = Anthropic()
    handler = ManagedAgentsHandler(nodeloom, agent_name="my-agent")

    session = anthropic_client.beta.sessions.create(agent=agent_id, environment_id=env_id)

    with handler.trace_session(session.id) as ctx:
        with anthropic_client.beta.sessions.events.stream(session.id) as stream:
            ctx.check_input(user_message)
            anthropic_client.beta.sessions.events.send(session.id, events=[
                {"type": "user.message", "content": [{"type": "text", "text": user_message}]}
            ])
            for event in stream:
                ctx.on_event(event)
                if event.type == "session.status_idle":
                    break
"""

from contextlib import contextmanager
from typing import Any, Optional


class ManagedAgentsHandler:
    """Handler for auto-instrumenting Anthropic Managed Agent sessions."""

    def __init__(self, client, agent_name: str = "anthropic-managed-agent",
                 agent_version: Optional[str] = None, guardrails: bool = True):
        self._client = client
        self._agent_name = agent_name
        self._agent_version = agent_version
        self._guardrails = guardrails

    @contextmanager
    def trace_session(self, session_id: str, **kwargs):
        """Context manager that creates a trace for an Anthropic Managed Agent session.

        Args:
            session_id: The Anthropic session ID.
            **kwargs: Additional kwargs passed to the trace (e.g., input, metadata).

        Yields:
            SessionContext with on_event(), check_input(), check_output() methods.
        """
        trace = self._client.trace(
            self._agent_name,
            session_id=session_id,
            **kwargs
        )
        if self._agent_version:
            trace._agent_version = self._agent_version

        ctx = _SessionContext(trace, self._client, self._guardrails)
        try:
            yield ctx
        except Exception as e:
            trace.end(status="error", output={"error": str(e)})
            raise
        else:
            trace.end(status="success", output=ctx._last_output)

    def check_input(self, text: str, **kwargs) -> dict:
        """Run guardrail checks on input text before sending to the agent.

        Returns:
            Guardrail result dict with 'passed', 'violations', 'redactedContent'.
        """
        return self._client.api.check_guardrails(
            text=text,
            detect_prompt_injection=True,
            redact_pii=True,
            **kwargs
        )

    def check_output(self, text: str, **kwargs) -> dict:
        """Run guardrail checks on agent output before showing to the user.

        Returns:
            Guardrail result dict with 'passed', 'violations', 'redactedContent'.
        """
        return self._client.api.check_guardrails(
            text=text,
            redact_pii=True,
            filter_content=True,
            **kwargs
        )


class _SessionContext:
    """Internal context for tracking events within a session trace."""

    def __init__(self, trace, client, guardrails: bool):
        self._trace = trace
        self._client = client
        self._guardrails = guardrails
        self._last_output = None
        self._active_spans = {}

    def on_event(self, event) -> None:
        """Process an Anthropic SSE event and create appropriate spans.

        Handles event types:
        - agent.message: Creates an LLM span with message content
        - agent.tool_use: Creates a TOOL span
        - agent.thinking: Creates a custom span for reasoning
        - session.status_idle: Finalizes the trace
        """
        event_type = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else None)
        if not event_type:
            return

        if event_type == "agent.message":
            self._handle_message(event)
        elif event_type == "agent.tool_use":
            self._handle_tool_use(event)
        elif event_type == "agent.thinking":
            self._handle_thinking(event)
        elif event_type == "agent.tool_result":
            self._handle_tool_result(event)

    def check_input(self, text: str, **kwargs) -> dict:
        """Run guardrail checks on input text."""
        if not self._guardrails:
            return {"passed": True, "violations": []}
        return self._client.api.check_guardrails(
            text=text,
            detect_prompt_injection=True,
            redact_pii=True,
            **kwargs
        )

    def check_output(self, text: str, **kwargs) -> dict:
        """Run guardrail checks on agent output."""
        if not self._guardrails:
            return {"passed": True, "violations": []}
        return self._client.api.check_guardrails(
            text=text,
            redact_pii=True,
            filter_content=True,
            **kwargs
        )

    def _handle_message(self, event):
        content = self._extract_text(event)
        span = self._trace.span("llm-response", span_type="llm")
        if content:
            span.set_output({"text": content})
            self._last_output = {"text": content}
            if self._guardrails:
                try:
                    result = self.check_output(content)
                    if not result.get("passed", True):
                        self._client.event("guardrail_violation", data={
                            "source": "anthropic-managed-agents",
                            "direction": "output",
                            "violations": result.get("violations", []),
                        })
                except Exception:
                    pass
        span.end()

    def _handle_tool_use(self, event):
        name = getattr(event, "name", None) or (event.get("name") if isinstance(event, dict) else "tool")
        tool_input = getattr(event, "input", None) or (event.get("input") if isinstance(event, dict) else None)
        span = self._trace.span(name, span_type="tool")
        if tool_input:
            span.set_input(tool_input)
        tool_id = getattr(event, "id", None) or (event.get("id") if isinstance(event, dict) else None)
        if tool_id:
            self._active_spans[tool_id] = span
        else:
            span.end()

    def _handle_tool_result(self, event):
        tool_id = getattr(event, "tool_use_id", None) or (event.get("tool_use_id") if isinstance(event, dict) else None)
        if tool_id and tool_id in self._active_spans:
            span = self._active_spans.pop(tool_id)
            content = self._extract_text(event)
            if content:
                span.set_output({"result": content})
            span.end()

    def _handle_thinking(self, event):
        content = self._extract_text(event)
        span = self._trace.span("thinking", span_type="custom")
        if content:
            span.set_input({"thinking": content})
        span.end()

    @staticmethod
    def _extract_text(event) -> Optional[str]:
        """Extract text content from an event, handling both object and dict formats."""
        # Object format (Anthropic SDK objects)
        content = getattr(event, "content", None)
        if content and isinstance(content, (list, tuple)):
            texts = []
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    texts.append(text)
            if texts:
                return " ".join(texts)

        # Dict format
        if isinstance(event, dict):
            content = event.get("content")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("text")]
                if texts:
                    return " ".join(texts)
            if isinstance(content, str):
                return content

        return None
