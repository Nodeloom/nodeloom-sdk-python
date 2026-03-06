"""Tests for Trace and Span lifecycle."""

import unittest
from unittest.mock import MagicMock, patch

from nodeloom.queue import TelemetryQueue
from nodeloom.span import Span
from nodeloom.trace import Trace
from nodeloom.types import SpanType, TraceStatus


class TestTraceCreation(unittest.TestCase):
    """Tests for trace initialization and start event emission."""

    def setUp(self):
        self.queue = MagicMock(spec=TelemetryQueue)

    def test_trace_has_unique_id(self):
        t1 = Trace("agent-a", queue=self.queue)
        t2 = Trace("agent-b", queue=self.queue)
        self.assertNotEqual(t1.trace_id, t2.trace_id)

    def test_trace_emits_start_event(self):
        trace = Trace("my-agent", queue=self.queue, input_data={"q": "hi"})

        self.queue.put.assert_called_once()
        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["type"], "trace_start")
        self.assertEqual(event["agent_name"], "my-agent")
        self.assertEqual(event["input"], {"q": "hi"})
        self.assertIn("trace_id", event)
        self.assertIn("timestamp", event)

    def test_trace_start_with_metadata(self):
        trace = Trace(
            "agent",
            queue=self.queue,
            agent_version="2.1",
            metadata={"team": "ml"},
        )

        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["agent_version"], "2.1")
        self.assertEqual(event["metadata"], {"team": "ml"})

    def test_trace_start_without_optional_fields(self):
        trace = Trace("agent", queue=self.queue)

        event = self.queue.put.call_args[0][0]
        self.assertNotIn("input", event)
        self.assertNotIn("agent_version", event)
        self.assertNotIn("metadata", event)

    def test_trace_properties(self):
        trace = Trace("test-agent", queue=self.queue)
        self.assertEqual(trace.agent_name, "test-agent")
        self.assertFalse(trace.ended)
        self.assertIsNotNone(trace.trace_id)


class TestTraceEnd(unittest.TestCase):
    """Tests for trace finalization."""

    def setUp(self):
        self.queue = MagicMock(spec=TelemetryQueue)

    def test_end_emits_trace_end_event(self):
        trace = Trace("agent", queue=self.queue)
        self.queue.reset_mock()

        trace.end(status=TraceStatus.SUCCESS, output={"result": "done"})

        self.queue.put.assert_called_once()
        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["type"], "trace_end")
        self.assertEqual(event["trace_id"], trace.trace_id)
        self.assertEqual(event["status"], "success")
        self.assertEqual(event["output"], {"result": "done"})
        self.assertIn("timestamp", event)

    def test_end_with_error(self):
        trace = Trace("agent", queue=self.queue)
        self.queue.reset_mock()

        trace.end(status=TraceStatus.ERROR, error="something broke")

        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error"], "something broke")

    def test_end_is_idempotent(self):
        trace = Trace("agent", queue=self.queue)
        self.queue.reset_mock()

        trace.end()
        trace.end()  # second call should be a no-op

        self.queue.put.assert_called_once()
        self.assertTrue(trace.ended)

    def test_end_defaults_to_success(self):
        trace = Trace("agent", queue=self.queue)
        self.queue.reset_mock()

        trace.end()

        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["status"], "success")


class TestTraceSpanCreation(unittest.TestCase):
    """Tests for creating spans from a trace."""

    def setUp(self):
        self.queue = MagicMock(spec=TelemetryQueue)

    def test_span_returns_span_instance(self):
        trace = Trace("agent", queue=self.queue)
        span = trace.span("my-operation")

        self.assertIsInstance(span, Span)
        self.assertEqual(span.name, "my-operation")
        self.assertEqual(span.trace_id, trace.trace_id)

    def test_span_with_type(self):
        trace = Trace("agent", queue=self.queue)
        span = trace.span("llm-call", type=SpanType.LLM)

        self.assertIsInstance(span, Span)

    def test_span_with_parent(self):
        trace = Trace("agent", queue=self.queue)
        parent = trace.span("parent")
        child = trace.span("child", parent_span_id=parent.span_id)

        self.assertEqual(child.parent_span_id, parent.span_id)

    def test_span_on_ended_trace_logs_warning(self):
        trace = Trace("agent", queue=self.queue)
        trace.end()

        with self.assertLogs("nodeloom.trace", level="WARNING"):
            span = trace.span("late-span")

        # Should still return a span (fire and forget)
        self.assertIsInstance(span, Span)


class TestTraceEvent(unittest.TestCase):
    """Tests for standalone events on a trace."""

    def setUp(self):
        self.queue = MagicMock(spec=TelemetryQueue)

    def test_event_enqueued(self):
        trace = Trace("agent", queue=self.queue)
        self.queue.reset_mock()

        trace.event("guardrail_triggered", level="warn", data={"rule": "pii"})

        self.queue.put.assert_called_once()
        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["type"], "event")
        self.assertEqual(event["trace_id"], trace.trace_id)
        self.assertEqual(event["name"], "guardrail_triggered")
        self.assertEqual(event["level"], "warn")
        self.assertEqual(event["data"], {"rule": "pii"})


class TestTraceContextManager(unittest.TestCase):
    """Tests for using Trace as a context manager."""

    def setUp(self):
        self.queue = MagicMock(spec=TelemetryQueue)

    def test_context_manager_auto_ends_success(self):
        with Trace("agent", queue=self.queue) as trace:
            pass

        self.assertTrue(trace.ended)
        # Last put call should be trace_end with success
        last_event = self.queue.put.call_args_list[-1][0][0]
        self.assertEqual(last_event["type"], "trace_end")
        self.assertEqual(last_event["status"], "success")

    def test_context_manager_auto_ends_error_on_exception(self):
        with self.assertRaises(ValueError):
            with Trace("agent", queue=self.queue) as trace:
                raise ValueError("boom")

        self.assertTrue(trace.ended)
        last_event = self.queue.put.call_args_list[-1][0][0]
        self.assertEqual(last_event["type"], "trace_end")
        self.assertEqual(last_event["status"], "error")
        self.assertIn("ValueError: boom", last_event["error"])

    def test_context_manager_does_not_suppress_exception(self):
        with self.assertRaises(RuntimeError):
            with Trace("agent", queue=self.queue):
                raise RuntimeError("not suppressed")

    def test_manual_end_prevents_auto_end(self):
        with Trace("agent", queue=self.queue) as trace:
            trace.end(status=TraceStatus.SUCCESS, output={"x": 1})

        # Only one trace_end should be emitted (the manual one)
        end_events = [
            c[0][0]
            for c in self.queue.put.call_args_list
            if c[0][0].get("type") == "trace_end"
        ]
        self.assertEqual(len(end_events), 1)
        self.assertEqual(end_events[0]["output"], {"x": 1})


class TestSpanCreation(unittest.TestCase):
    """Tests for Span initialization."""

    def setUp(self):
        self.queue = MagicMock(spec=TelemetryQueue)

    def test_span_has_unique_id(self):
        s1 = Span("op1", trace_id="t1", queue=self.queue)
        s2 = Span("op2", trace_id="t1", queue=self.queue)
        self.assertNotEqual(s1.span_id, s2.span_id)

    def test_span_properties(self):
        span = Span(
            "my-op",
            trace_id="trace-123",
            queue=self.queue,
            span_type=SpanType.LLM,
            parent_span_id="parent-456",
        )
        self.assertEqual(span.name, "my-op")
        self.assertEqual(span.trace_id, "trace-123")
        self.assertEqual(span.parent_span_id, "parent-456")
        self.assertFalse(span.ended)


class TestSpanSetters(unittest.TestCase):
    """Tests for span data setters."""

    def setUp(self):
        self.queue = MagicMock(spec=TelemetryQueue)

    def test_set_input(self):
        span = Span("op", trace_id="t1", queue=self.queue)
        result = span.set_input({"prompt": "hello"})

        self.assertIs(result, span)  # returns self for chaining

    def test_set_output(self):
        span = Span("op", trace_id="t1", queue=self.queue)
        result = span.set_output({"response": "world"})

        self.assertIs(result, span)

    def test_set_error(self):
        span = Span("op", trace_id="t1", queue=self.queue)
        result = span.set_error("something failed")

        self.assertIs(result, span)

    def test_set_token_usage(self):
        span = Span("op", trace_id="t1", queue=self.queue)
        result = span.set_token_usage(prompt=100, completion=200, model="gpt-4o")

        self.assertIs(result, span)

    def test_set_token_usage_calculates_total(self):
        span = Span("op", trace_id="t1", queue=self.queue)
        span.set_token_usage(prompt=100, completion=200, model="gpt-4o")
        span.end()

        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["token_usage"]["prompt_tokens"], 100)
        self.assertEqual(event["token_usage"]["completion_tokens"], 200)
        self.assertEqual(event["token_usage"]["total_tokens"], 300)
        self.assertEqual(event["token_usage"]["model"], "gpt-4o")

    def test_set_token_usage_without_model(self):
        span = Span("op", trace_id="t1", queue=self.queue)
        span.set_token_usage(prompt=50, completion=75)
        span.end()

        event = self.queue.put.call_args[0][0]
        self.assertNotIn("model", event["token_usage"])


class TestSpanEnd(unittest.TestCase):
    """Tests for span finalization."""

    def setUp(self):
        self.queue = MagicMock(spec=TelemetryQueue)

    def test_end_emits_span_event(self):
        span = Span("op", trace_id="t1", queue=self.queue, span_type=SpanType.LLM)
        span.set_input({"prompt": "hi"})
        span.set_output({"text": "hello"})
        span.end()

        self.queue.put.assert_called_once()
        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["type"], "span")
        self.assertEqual(event["trace_id"], "t1")
        self.assertEqual(event["name"], "op")
        self.assertEqual(event["span_type"], "llm")
        self.assertEqual(event["status"], "success")
        self.assertEqual(event["input"], {"prompt": "hi"})
        self.assertEqual(event["output"], {"text": "hello"})
        self.assertIn("timestamp", event)
        self.assertIn("end_timestamp", event)
        self.assertIsNone(event["parent_span_id"])

    def test_end_with_error(self):
        span = Span("op", trace_id="t1", queue=self.queue)
        span.set_error("failure")
        span.end()

        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error"], "failure")

    def test_end_with_override_status(self):
        span = Span("op", trace_id="t1", queue=self.queue)
        span.end(status=TraceStatus.ERROR, output={"partial": True})

        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["output"], {"partial": True})

    def test_end_is_idempotent(self):
        span = Span("op", trace_id="t1", queue=self.queue)
        span.end()

        with self.assertLogs("nodeloom.span", level="WARNING"):
            span.end()

        # Only one event should be enqueued
        self.queue.put.assert_called_once()
        self.assertTrue(span.ended)

    def test_end_omits_none_fields(self):
        span = Span("op", trace_id="t1", queue=self.queue)
        span.end()

        event = self.queue.put.call_args[0][0]
        self.assertNotIn("input", event)
        self.assertNotIn("output", event)
        self.assertNotIn("error", event)
        self.assertNotIn("token_usage", event)


class TestSpanContextManager(unittest.TestCase):
    """Tests for using Span as a context manager."""

    def setUp(self):
        self.queue = MagicMock(spec=TelemetryQueue)

    def test_context_manager_auto_ends_success(self):
        with Span("op", trace_id="t1", queue=self.queue) as span:
            span.set_output({"data": "value"})

        self.assertTrue(span.ended)
        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["status"], "success")

    def test_context_manager_auto_ends_error_on_exception(self):
        with self.assertRaises(TypeError):
            with Span("op", trace_id="t1", queue=self.queue) as span:
                raise TypeError("bad type")

        self.assertTrue(span.ended)
        event = self.queue.put.call_args[0][0]
        self.assertEqual(event["status"], "error")
        self.assertIn("TypeError: bad type", event["error"])

    def test_context_manager_does_not_suppress(self):
        with self.assertRaises(KeyError):
            with Span("op", trace_id="t1", queue=self.queue):
                raise KeyError("missing")

    def test_manual_end_prevents_auto_end(self):
        with Span("op", trace_id="t1", queue=self.queue) as span:
            span.end(status=TraceStatus.SUCCESS)

        # Should only have one put call
        self.queue.put.assert_called_once()


class TestNestedSpans(unittest.TestCase):
    """Tests for nested trace/span workflows."""

    def setUp(self):
        self.queue = MagicMock(spec=TelemetryQueue)

    def test_full_trace_lifecycle(self):
        """Simulate a complete trace with nested spans."""
        trace = Trace("agent", queue=self.queue, input_data={"query": "test"})
        self.queue.reset_mock()

        with trace.span("chain", type=SpanType.CHAIN) as chain_span:
            chain_span.set_input({"query": "test"})
            with trace.span(
                "llm-call", type=SpanType.LLM, parent_span_id=chain_span.span_id
            ) as llm_span:
                llm_span.set_input({"prompt": "test"})
                llm_span.set_output({"text": "response"})
                llm_span.set_token_usage(prompt=10, completion=20, model="gpt-4o")

            chain_span.set_output({"result": "response"})

        trace.end(status=TraceStatus.SUCCESS, output={"result": "response"})

        # Should have: llm span, chain span, trace_end = 3 events
        self.assertEqual(self.queue.put.call_count, 3)

        events = [c[0][0] for c in self.queue.put.call_args_list]

        # First: LLM span (inner span ends first)
        self.assertEqual(events[0]["type"], "span")
        self.assertEqual(events[0]["span_type"], "llm")
        self.assertEqual(events[0]["parent_span_id"], chain_span.span_id)

        # Second: chain span
        self.assertEqual(events[1]["type"], "span")
        self.assertEqual(events[1]["span_type"], "chain")

        # Third: trace_end
        self.assertEqual(events[2]["type"], "trace_end")
        self.assertEqual(events[2]["status"], "success")


if __name__ == "__main__":
    unittest.main()
