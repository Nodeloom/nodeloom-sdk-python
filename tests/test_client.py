"""Tests for NodeLoomClient."""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from nodeloom.client import NodeLoomClient
from nodeloom.config import NodeLoomConfig
from nodeloom.trace import Trace
from nodeloom.types import EventLevel


class TestNodeLoomClientInit(unittest.TestCase):
    """Tests for client initialization."""

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_init_with_defaults(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_test123")

        self.assertEqual(client.config.api_key, "sdk_test123")
        self.assertEqual(client.config.endpoint, "https://api.nodeloom.io")
        self.assertEqual(client.config.environment, "production")
        self.assertEqual(client.config.batch_size, 100)
        self.assertEqual(client.config.flush_interval, 5.0)
        self.assertEqual(client.config.max_retries, 3)
        self.assertTrue(client.is_enabled)
        mock_processor_cls.return_value.start.assert_called_once()

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_init_custom_config(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(
            api_key="sdk_custom",
            endpoint="https://custom.nodeloom.io",
            environment="staging",
            batch_size=50,
            flush_interval=2.0,
            max_retries=5,
            queue_max_size=5000,
            timeout=30.0,
        )

        self.assertEqual(client.config.endpoint, "https://custom.nodeloom.io")
        self.assertEqual(client.config.environment, "staging")
        self.assertEqual(client.config.batch_size, 50)
        self.assertEqual(client.config.flush_interval, 2.0)
        self.assertEqual(client.config.max_retries, 5)
        self.assertEqual(client.config.queue_max_size, 5000)
        self.assertEqual(client.config.timeout, 30.0)

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_init_disabled(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_disabled", enabled=False)

        self.assertFalse(client.is_enabled)
        mock_processor_cls.return_value.start.assert_not_called()

    def test_init_empty_api_key_raises(self):
        with self.assertRaises(ValueError):
            NodeLoomClient(api_key="")

    def test_init_invalid_batch_size_raises(self):
        with self.assertRaises(ValueError):
            NodeLoomClient(api_key="sdk_test", batch_size=0)

    def test_init_invalid_flush_interval_raises(self):
        with self.assertRaises(ValueError):
            NodeLoomClient(api_key="sdk_test", flush_interval=-1)


class TestNodeLoomClientTrace(unittest.TestCase):
    """Tests for trace creation."""

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_trace_returns_trace_object(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_test")
        trace = client.trace("my-agent")

        self.assertIsInstance(trace, Trace)
        self.assertEqual(trace.agent_name, "my-agent")

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_trace_with_input(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_test")
        trace = client.trace("agent", input={"query": "hello"})

        self.assertIsInstance(trace, Trace)

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_trace_with_metadata(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_test")
        trace = client.trace(
            "agent",
            input={"q": "test"},
            agent_version="1.0.0",
            metadata={"env": "test"},
        )

        self.assertIsInstance(trace, Trace)

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_trace_emits_start_event(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        """Verify that creating a trace immediately enqueues a trace_start event."""
        mock_queue_instance = mock_queue_cls.return_value
        client = NodeLoomClient(api_key="sdk_test")
        trace = client.trace("my-agent", input={"x": 1})

        # The queue's put() should have been called with a trace_start event
        calls = mock_queue_instance.put.call_args_list
        self.assertTrue(len(calls) >= 1)
        event = calls[0][0][0]
        self.assertEqual(event["type"], "trace_start")
        self.assertEqual(event["agent_name"], "my-agent")
        self.assertEqual(event["input"], {"x": 1})


class TestNodeLoomClientEvent(unittest.TestCase):
    """Tests for standalone events."""

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_event_enqueued(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        mock_queue_instance = mock_queue_cls.return_value
        client = NodeLoomClient(api_key="sdk_test")

        client.event(
            name="guardrail_triggered",
            level=EventLevel.WARN,
            data={"rule": "pii"},
            trace_id="some-trace-id",
        )

        calls = mock_queue_instance.put.call_args_list
        # Find the event call (there might be others)
        event_calls = [c for c in calls if c[0][0].get("type") == "event"]
        self.assertEqual(len(event_calls), 1)
        evt = event_calls[0][0][0]
        self.assertEqual(evt["name"], "guardrail_triggered")
        self.assertEqual(evt["level"], "warn")
        self.assertEqual(evt["data"], {"rule": "pii"})
        self.assertEqual(evt["trace_id"], "some-trace-id")

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_event_without_trace_id(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        mock_queue_instance = mock_queue_cls.return_value
        client = NodeLoomClient(api_key="sdk_test")

        client.event(name="startup", level=EventLevel.INFO)

        event_calls = [
            c for c in mock_queue_instance.put.call_args_list
            if c[0][0].get("type") == "event"
        ]
        self.assertEqual(len(event_calls), 1)
        evt = event_calls[0][0][0]
        self.assertIsNone(evt["trace_id"])


class TestNodeLoomClientFlush(unittest.TestCase):
    """Tests for flush behavior."""

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_flush_delegates_to_processor(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_test")
        client.flush()

        mock_processor_cls.return_value.flush.assert_called_once()

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_flush_noop_when_disabled(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_test", enabled=False)
        client.flush()

        mock_processor_cls.return_value.flush.assert_not_called()


class TestNodeLoomClientShutdown(unittest.TestCase):
    """Tests for shutdown behavior."""

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_shutdown_stops_processor(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_test")
        client.shutdown()

        mock_processor_cls.return_value.shutdown.assert_called_once_with(timeout=10.0)
        mock_transport_cls.return_value.close.assert_called_once()

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_shutdown_custom_timeout(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_test")
        client.shutdown(timeout=30.0)

        mock_processor_cls.return_value.shutdown.assert_called_once_with(timeout=30.0)

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_shutdown_idempotent(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_test")
        client.shutdown()
        client.shutdown()

        # shutdown should only be called once on the processor
        mock_processor_cls.return_value.shutdown.assert_called_once()

    @patch("nodeloom.client.BatchProcessor")
    @patch("nodeloom.client.HttpTransport")
    @patch("nodeloom.client.TelemetryQueue")
    def test_shutdown_when_disabled(self, mock_queue_cls, mock_transport_cls, mock_processor_cls):
        client = NodeLoomClient(api_key="sdk_test", enabled=False)
        client.shutdown()

        mock_processor_cls.return_value.shutdown.assert_not_called()
        mock_transport_cls.return_value.close.assert_called_once()


class TestNodeLoomImport(unittest.TestCase):
    """Test that the convenience alias works."""

    def test_nodeloom_alias(self):
        from nodeloom import NodeLoom

        self.assertIs(NodeLoom, NodeLoomClient)

    def test_public_exports(self):
        import nodeloom

        self.assertTrue(hasattr(nodeloom, "NodeLoom"))
        self.assertTrue(hasattr(nodeloom, "Trace"))
        self.assertTrue(hasattr(nodeloom, "Span"))
        self.assertTrue(hasattr(nodeloom, "SpanType"))
        self.assertTrue(hasattr(nodeloom, "TraceStatus"))
        self.assertTrue(hasattr(nodeloom, "EventLevel"))
        self.assertTrue(hasattr(nodeloom, "__version__"))


if __name__ == "__main__":
    unittest.main()
