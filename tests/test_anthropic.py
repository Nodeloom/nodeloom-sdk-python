"""Tests for Anthropic Managed Agents integration."""
import pytest
from unittest.mock import MagicMock, patch

from nodeloom import NodeLoom
from nodeloom.api import ApiClient
from nodeloom.integrations.anthropic import ManagedAgentsHandler


class MockEvent:
    def __init__(self, type, **kwargs):
        self.type = type
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockContentBlock:
    def __init__(self, text):
        self.text = text


def make_client():
    client = MagicMock()
    client.trace.return_value.__enter__ = MagicMock()
    client.trace.return_value.__exit__ = MagicMock(return_value=False)
    # Make trace return a mock that supports span()
    trace = MagicMock()
    span = MagicMock()
    trace.span.return_value = span
    client.trace.return_value = trace
    client.api.check_guardrails.return_value = {"passed": True, "violations": []}
    return client, trace, span


class TestManagedAgentsHandler:
    def test_creates_handler(self):
        client = MagicMock()
        handler = ManagedAgentsHandler(client, agent_name="test-agent")
        assert handler._agent_name == "test-agent"
        assert handler._guardrails is True

    def test_creates_handler_without_guardrails(self):
        client = MagicMock()
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)
        assert handler._guardrails is False

    def test_check_input_calls_guardrails(self):
        client = MagicMock()
        client.api.check_guardrails.return_value = {"passed": True, "violations": []}
        handler = ManagedAgentsHandler(client, agent_name="test")
        result = handler.check_input("hello world")
        client.api.check_guardrails.assert_called_once()
        assert result["passed"] is True

    def test_check_output_calls_guardrails(self):
        client = MagicMock()
        client.api.check_guardrails.return_value = {"passed": False, "violations": [{"type": "PII"}]}
        handler = ManagedAgentsHandler(client, agent_name="test")
        result = handler.check_output("my SSN is 123-45-6789")
        assert result["passed"] is False


class TestSessionContext:
    def test_on_event_handles_agent_message(self):
        client, trace, span = make_client()
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)

        with handler.trace_session("sess_123") as ctx:
            event = MockEvent("agent.message", content=[MockContentBlock("Hello!")])
            ctx.on_event(event)

        trace.span.assert_called_with("llm-response", span_type="llm")
        span.set_output.assert_called_with({"text": "Hello!"})
        span.end.assert_called()

    def test_on_event_handles_tool_use(self):
        client, trace, span = make_client()
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)

        with handler.trace_session("sess_123") as ctx:
            event = MockEvent("agent.tool_use", name="bash", input={"command": "ls"}, id=None)
            ctx.on_event(event)

        trace.span.assert_called_with("bash", span_type="tool")
        span.set_input.assert_called_with({"command": "ls"})

    def test_on_event_handles_thinking(self):
        client, trace, span = make_client()
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)

        with handler.trace_session("sess_123") as ctx:
            event = MockEvent("agent.thinking", content=[MockContentBlock("Let me think...")])
            ctx.on_event(event)

        trace.span.assert_called_with("thinking", span_type="custom")

    def test_on_event_ignores_unknown_types(self):
        client, trace, span = make_client()
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)

        with handler.trace_session("sess_123") as ctx:
            event = MockEvent("unknown.type")
            ctx.on_event(event)

        trace.span.assert_not_called()

    def test_guardrail_check_on_output(self):
        client, trace, span = make_client()
        client.api.check_guardrails.return_value = {"passed": False, "violations": [{"type": "PII"}]}
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=True)

        with handler.trace_session("sess_123") as ctx:
            event = MockEvent("agent.message", content=[MockContentBlock("SSN: 123-45-6789")])
            ctx.on_event(event)

        client.event.assert_called_once()

    def test_context_handles_exception(self):
        client, trace, span = make_client()
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)

        with pytest.raises(ValueError):
            with handler.trace_session("sess_123") as ctx:
                raise ValueError("test error")

        trace.end.assert_called_with(status="error", output={"error": "test error"})

    def test_dict_event_format(self):
        client, trace, span = make_client()
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)

        with handler.trace_session("sess_123") as ctx:
            event = {"type": "agent.tool_use", "name": "web_search", "input": {"query": "test"}, "id": None}
            ctx.on_event(event)

        trace.span.assert_called_with("web_search", span_type="tool")

    def test_tool_result_closes_active_span(self):
        client, trace, _ = make_client()
        tool_span = MagicMock()
        trace.span.return_value = tool_span
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)

        with handler.trace_session("sess_123") as ctx:
            tool_event = MockEvent("agent.tool_use", name="bash", input={"cmd": "ls"}, id="tool_1")
            ctx.on_event(tool_event)
            result_event = MockEvent("agent.tool_result", tool_use_id="tool_1", content="file1.txt")
            ctx.on_event(result_event)

        # span.end() called once for tool_result (not during tool_use since it has an id)
        assert tool_span.end.call_count >= 1

    def test_trace_ends_with_success(self):
        client, trace, span = make_client()
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)

        with handler.trace_session("sess_123") as ctx:
            event = MockEvent("agent.message", content=[MockContentBlock("Done!")])
            ctx.on_event(event)

        trace.end.assert_called_with(status="success", output={"text": "Done!"})

    def test_agent_version_set_on_trace(self):
        client, trace, span = make_client()
        handler = ManagedAgentsHandler(client, agent_name="test", agent_version="1.2.3", guardrails=False)

        with handler.trace_session("sess_123") as ctx:
            pass

        assert trace._agent_version == "1.2.3"

    def test_context_check_input_skips_when_guardrails_disabled(self):
        client, trace, span = make_client()
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)

        with handler.trace_session("sess_123") as ctx:
            result = ctx.check_input("hello")

        assert result == {"passed": True, "violations": []}
        client.api.check_guardrails.assert_not_called()

    def test_extract_text_with_dict_string_content(self):
        client, trace, span = make_client()
        handler = ManagedAgentsHandler(client, agent_name="test", guardrails=False)

        with handler.trace_session("sess_123") as ctx:
            event = {"type": "agent.message", "content": "plain text response"}
            ctx.on_event(event)

        span.set_output.assert_called_with({"text": "plain text response"})


class TestIntegrationWithRealApiClient:
    """Exercises the Anthropic handler through the real ApiClient so that
    signature drift between the handler and the HTTP layer can't silently
    regress. A prior version of the handler omitted team_id from the
    check_guardrails call and raised TypeError at runtime — mocked tests
    never caught it."""

    @patch("nodeloom.api.requests.Session")
    def test_check_input_reaches_backend_with_agent_name(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_response = MagicMock(ok=True, status_code=200)
        mock_response.json.return_value = {
            "passed": True,
            "violations": [],
            "redactedContent": None,
            "checks": [],
            "guardrailSessionId": "sess-abc",
        }
        mock_session.request.return_value = mock_response

        # Minimal client that routes api() to a real ApiClient.
        client = MagicMock()
        client.api = ApiClient(api_key="sdk_test", endpoint="https://example.com")

        handler = ManagedAgentsHandler(client, agent_name="anthropic-agent")
        # The real call path must not raise even though team_id is omitted.
        result = handler.check_input("hello world")
        assert result["passed"] is True

        # Verify the HTTP body carried the handler's agent_name so the backend
        # can bind the guardrail session to that agent for HARD-mode checks.
        body = mock_session.request.call_args.kwargs["json"]
        assert body["agentName"] == "anthropic-agent"
        # No team_id → no teamId query param; backend infers from SDK token.
        params = mock_session.request.call_args.kwargs.get("params")
        assert params is None or "teamId" not in (params or {})
