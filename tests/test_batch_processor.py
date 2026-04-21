"""Tests for BatchProcessor, TelemetryQueue, and HttpTransport."""

import json
import time
import threading
import unittest
from unittest.mock import MagicMock, patch, call

from nodeloom.batch_processor import BatchProcessor
from nodeloom.config import NodeLoomConfig
from nodeloom.queue import TelemetryQueue
from nodeloom.transport import HttpTransport


class TestTelemetryQueue(unittest.TestCase):
    """Tests for the bounded event queue."""

    def test_put_and_drain(self):
        q = TelemetryQueue(max_size=100)
        q.put({"type": "span", "id": 1})
        q.put({"type": "span", "id": 2})

        items = q.drain(10)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], 1)
        self.assertEqual(items[1]["id"], 2)

    def test_drain_respects_max_items(self):
        q = TelemetryQueue(max_size=100)
        for i in range(10):
            q.put({"id": i})

        items = q.drain(3)
        self.assertEqual(len(items), 3)

        remaining = q.drain(100)
        self.assertEqual(len(remaining), 7)

    def test_drain_empty_queue(self):
        q = TelemetryQueue(max_size=100)
        items = q.drain(10)
        self.assertEqual(items, [])

    def test_queue_full_drops_event(self):
        q = TelemetryQueue(max_size=3)
        self.assertTrue(q.put({"id": 1}))
        self.assertTrue(q.put({"id": 2}))
        self.assertTrue(q.put({"id": 3}))
        self.assertFalse(q.put({"id": 4}))  # dropped

        self.assertEqual(q.dropped_count, 1)
        self.assertEqual(q.size(), 3)

    def test_size_and_is_empty(self):
        q = TelemetryQueue(max_size=100)
        self.assertTrue(q.is_empty())
        self.assertEqual(q.size(), 0)

        q.put({"id": 1})
        self.assertFalse(q.is_empty())
        self.assertEqual(q.size(), 1)


class TestHttpTransport(unittest.TestCase):
    """Tests for HTTP transport with retry logic."""

    def _make_config(self, **kwargs):
        defaults = {
            "api_key": "sdk_test",
            "endpoint": "https://api.nodeloom.io",
            "max_retries": 3,
            "timeout": 5.0,
        }
        defaults.update(kwargs)
        return NodeLoomConfig(**defaults)

    @patch("nodeloom.transport.requests.Session")
    def test_successful_send(self, mock_session_cls):
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"accepted": 2, "rejected": 0, "errors": []}
        mock_session.post.return_value = mock_response

        config = self._make_config()
        transport = HttpTransport(config)
        result = transport.send_batch([{"type": "span"}, {"type": "span"}])

        self.assertIsNotNone(result)
        self.assertEqual(result["accepted"], 2)
        mock_session.post.assert_called_once()

        # Verify the payload structure
        call_kwargs = mock_session.post.call_args
        payload = call_kwargs[1]["json"]
        self.assertEqual(len(payload["events"]), 2)
        self.assertEqual(payload["sdk_version"], "0.8.0")
        self.assertEqual(payload["sdk_language"], "python")

    @patch("nodeloom.transport.requests.Session")
    def test_url_construction(self, mock_session_cls):
        config = self._make_config(endpoint="https://custom.example.com/")
        transport = HttpTransport(config)
        self.assertEqual(
            transport.url, "https://custom.example.com/api/sdk/v1/telemetry"
        )

    @patch("nodeloom.transport.requests.Session")
    def test_url_construction_no_trailing_slash(self, mock_session_cls):
        config = self._make_config(endpoint="https://custom.example.com")
        transport = HttpTransport(config)
        self.assertEqual(
            transport.url, "https://custom.example.com/api/sdk/v1/telemetry"
        )

    @patch("nodeloom.transport.requests.Session")
    def test_empty_batch_returns_none(self, mock_session_cls):
        config = self._make_config()
        transport = HttpTransport(config)
        result = transport.send_batch([])

        self.assertIsNone(result)
        mock_session_cls.return_value.post.assert_not_called()

    @patch("nodeloom.transport.time.sleep")
    @patch("nodeloom.transport.requests.Session")
    def test_retry_on_server_error(self, mock_session_cls, mock_sleep):
        mock_session = mock_session_cls.return_value
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_fail.text = "Internal Server Error"
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = {"accepted": 1, "rejected": 0, "errors": []}

        mock_session.post.side_effect = [mock_fail, mock_success]

        config = self._make_config(max_retries=3)
        transport = HttpTransport(config)
        result = transport.send_batch([{"type": "span"}])

        self.assertIsNotNone(result)
        self.assertEqual(mock_session.post.call_count, 2)
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1 second backoff

    @patch("nodeloom.transport.time.sleep")
    @patch("nodeloom.transport.requests.Session")
    def test_no_retry_on_client_error(self, mock_session_cls, mock_sleep):
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_session.post.return_value = mock_response

        config = self._make_config(max_retries=3)
        transport = HttpTransport(config)
        result = transport.send_batch([{"type": "span"}])

        self.assertIsNone(result)
        self.assertEqual(mock_session.post.call_count, 1)  # No retries
        mock_sleep.assert_not_called()

    @patch("nodeloom.transport.time.sleep")
    @patch("nodeloom.transport.requests.Session")
    def test_retry_on_connection_error(self, mock_session_cls, mock_sleep):
        import requests as req

        mock_session = mock_session_cls.return_value
        mock_session.post.side_effect = [
            req.ConnectionError("Connection refused"),
            req.ConnectionError("Connection refused"),
            MagicMock(
                status_code=200,
                json=MagicMock(
                    return_value={"accepted": 1, "rejected": 0, "errors": []}
                ),
            ),
        ]

        config = self._make_config(max_retries=3)
        transport = HttpTransport(config)
        result = transport.send_batch([{"type": "span"}])

        self.assertIsNotNone(result)
        self.assertEqual(mock_session.post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)  # 2^0=1, 2^1=2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch("nodeloom.transport.time.sleep")
    @patch("nodeloom.transport.requests.Session")
    def test_all_retries_exhausted(self, mock_session_cls, mock_sleep):
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        mock_session.post.return_value = mock_response

        config = self._make_config(max_retries=2)
        transport = HttpTransport(config)
        result = transport.send_batch([{"type": "span"}])

        self.assertIsNone(result)
        self.assertEqual(mock_session.post.call_count, 3)  # initial + 2 retries
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("nodeloom.transport.requests.Session")
    def test_partial_rejection_logged(self, mock_session_cls):
        mock_session = mock_session_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "accepted": 8,
            "rejected": 2,
            "errors": [
                {"index": 3, "error": "invalid span_type"},
                {"index": 7, "error": "missing trace_id"},
            ],
        }
        mock_session.post.return_value = mock_response

        config = self._make_config()
        transport = HttpTransport(config)

        with self.assertLogs("nodeloom.transport", level="WARNING") as cm:
            result = transport.send_batch([{"type": "span"}] * 10)

        self.assertIsNotNone(result)
        self.assertEqual(result["rejected"], 2)
        # Verify warning was logged
        self.assertTrue(any("partially rejected" in msg for msg in cm.output))

    @patch("nodeloom.transport.requests.Session")
    def test_close(self, mock_session_cls):
        config = self._make_config()
        transport = HttpTransport(config)
        transport.close()

        mock_session_cls.return_value.close.assert_called_once()

    @patch("nodeloom.transport.requests.Session")
    def test_auth_header(self, mock_session_cls):
        mock_session = mock_session_cls.return_value
        config = self._make_config(api_key="sdk_mykey123")
        transport = HttpTransport(config)

        # Check headers were set on session
        mock_session.headers.update.assert_called_once()
        headers = mock_session.headers.update.call_args[0][0]
        self.assertEqual(headers["Authorization"], "Bearer sdk_mykey123")
        self.assertEqual(headers["Content-Type"], "application/json")


class TestBatchProcessor(unittest.TestCase):
    """Tests for the background batch processing thread."""

    def _make_config(self, **kwargs):
        defaults = {
            "api_key": "sdk_test",
            "batch_size": 5,
            "flush_interval": 0.1,  # short for tests
            "max_retries": 0,
        }
        defaults.update(kwargs)
        return NodeLoomConfig(**defaults)

    def test_start_creates_daemon_thread(self):
        config = self._make_config()
        queue = TelemetryQueue()
        transport = MagicMock(spec=HttpTransport)

        processor = BatchProcessor(config, queue, transport)
        processor.start()

        try:
            # Give thread a moment to start
            time.sleep(0.05)
            self.assertTrue(processor._thread.is_alive())
            self.assertTrue(processor._thread.daemon)
            self.assertEqual(processor._thread.name, "nodeloom-batch-processor")
        finally:
            processor.shutdown(timeout=2.0)

    def test_flush_on_interval(self):
        config = self._make_config(flush_interval=0.1, batch_size=1000)
        queue = TelemetryQueue()
        transport = MagicMock(spec=HttpTransport)
        transport.send_batch.return_value = {"accepted": 3, "rejected": 0, "errors": []}

        # Add events before starting
        for i in range(3):
            queue.put({"type": "span", "id": i})

        processor = BatchProcessor(config, queue, transport)
        processor.start()

        try:
            # Wait for the flush interval to trigger
            time.sleep(0.3)
            transport.send_batch.assert_called()
            events = transport.send_batch.call_args[0][0]
            self.assertEqual(len(events), 3)
        finally:
            processor.shutdown(timeout=2.0)

    def test_flush_on_batch_size(self):
        config = self._make_config(batch_size=3, flush_interval=10.0)
        queue = TelemetryQueue()
        transport = MagicMock(spec=HttpTransport)
        transport.send_batch.return_value = {"accepted": 3, "rejected": 0, "errors": []}

        processor = BatchProcessor(config, queue, transport)
        processor.start()

        try:
            # Add enough events to trigger batch size flush
            for i in range(3):
                queue.put({"type": "span", "id": i})

            # Manually trigger flush since batch_size check is done at flush time
            processor.flush()
            time.sleep(0.2)

            transport.send_batch.assert_called()
            events = transport.send_batch.call_args[0][0]
            self.assertEqual(len(events), 3)
        finally:
            processor.shutdown(timeout=2.0)

    def test_shutdown_flushes_remaining(self):
        config = self._make_config(flush_interval=60.0)  # long interval
        queue = TelemetryQueue()
        transport = MagicMock(spec=HttpTransport)
        transport.send_batch.return_value = {"accepted": 2, "rejected": 0, "errors": []}

        processor = BatchProcessor(config, queue, transport)
        processor.start()

        # Add events after start
        queue.put({"type": "span", "id": 1})
        queue.put({"type": "span", "id": 2})

        # Shutdown should flush
        processor.shutdown(timeout=5.0)

        transport.send_batch.assert_called()
        events = transport.send_batch.call_args[0][0]
        self.assertEqual(len(events), 2)

    def test_shutdown_waits_for_thread(self):
        config = self._make_config(flush_interval=0.05)
        queue = TelemetryQueue()
        transport = MagicMock(spec=HttpTransport)

        processor = BatchProcessor(config, queue, transport)
        processor.start()

        processor.shutdown(timeout=5.0)

        self.assertFalse(processor._thread.is_alive())

    def test_multiple_start_calls_idempotent(self):
        config = self._make_config()
        queue = TelemetryQueue()
        transport = MagicMock(spec=HttpTransport)

        processor = BatchProcessor(config, queue, transport)
        processor.start()
        thread1 = processor._thread
        processor.start()  # should be no-op
        thread2 = processor._thread

        try:
            self.assertIs(thread1, thread2)
        finally:
            processor.shutdown(timeout=2.0)

    def test_multiple_batches(self):
        config = self._make_config(batch_size=2, flush_interval=0.05)
        queue = TelemetryQueue()
        transport = MagicMock(spec=HttpTransport)
        transport.send_batch.return_value = {"accepted": 2, "rejected": 0, "errors": []}

        # Add 5 events
        for i in range(5):
            queue.put({"type": "span", "id": i})

        processor = BatchProcessor(config, queue, transport)
        processor.start()

        try:
            time.sleep(0.3)
            # Should have been called multiple times (3 batches: 2+2+1)
            self.assertGreaterEqual(transport.send_batch.call_count, 2)
        finally:
            processor.shutdown(timeout=2.0)

    def test_transport_error_does_not_crash_processor(self):
        config = self._make_config(flush_interval=0.05)
        queue = TelemetryQueue()
        transport = MagicMock(spec=HttpTransport)
        transport.send_batch.side_effect = RuntimeError("network down")

        queue.put({"type": "span"})

        processor = BatchProcessor(config, queue, transport)
        processor.start()

        try:
            time.sleep(0.2)
            # Processor should still be alive despite the error
            self.assertTrue(processor._thread.is_alive())
        finally:
            processor.shutdown(timeout=2.0)

    def test_explicit_flush_wakes_processor(self):
        config = self._make_config(flush_interval=60.0)  # very long
        queue = TelemetryQueue()
        transport = MagicMock(spec=HttpTransport)
        transport.send_batch.return_value = {"accepted": 1, "rejected": 0, "errors": []}

        processor = BatchProcessor(config, queue, transport)
        processor.start()

        try:
            queue.put({"type": "span"})
            processor.flush()
            time.sleep(0.2)
            transport.send_batch.assert_called()
        finally:
            processor.shutdown(timeout=2.0)


class TestConfigValidation(unittest.TestCase):
    """Tests for NodeLoomConfig validation."""

    def test_valid_config(self):
        config = NodeLoomConfig(api_key="sdk_valid")
        self.assertEqual(config.api_key, "sdk_valid")
        self.assertEqual(config.endpoint, "https://api.nodeloom.io")

    def test_empty_api_key(self):
        with self.assertRaises(ValueError):
            NodeLoomConfig(api_key="")

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            NodeLoomConfig(api_key="sdk_test", batch_size=0)

    def test_negative_batch_size(self):
        with self.assertRaises(ValueError):
            NodeLoomConfig(api_key="sdk_test", batch_size=-1)

    def test_invalid_flush_interval(self):
        with self.assertRaises(ValueError):
            NodeLoomConfig(api_key="sdk_test", flush_interval=0)

    def test_negative_max_retries(self):
        with self.assertRaises(ValueError):
            NodeLoomConfig(api_key="sdk_test", max_retries=-1)

    def test_zero_max_retries_allowed(self):
        config = NodeLoomConfig(api_key="sdk_test", max_retries=0)
        self.assertEqual(config.max_retries, 0)

    def test_invalid_queue_max_size(self):
        with self.assertRaises(ValueError):
            NodeLoomConfig(api_key="sdk_test", queue_max_size=0)

    def test_invalid_timeout(self):
        with self.assertRaises(ValueError):
            NodeLoomConfig(api_key="sdk_test", timeout=0)

    def test_config_is_frozen(self):
        config = NodeLoomConfig(api_key="sdk_test")
        with self.assertRaises(AttributeError):
            config.api_key = "sdk_changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
