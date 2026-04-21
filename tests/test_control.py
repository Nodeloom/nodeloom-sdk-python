"""Tests for the SDK remote-control state machine."""

import time
import unittest
from unittest.mock import MagicMock, patch

from nodeloom import AgentHaltedError, NodeLoom
from nodeloom.api import ApiClient
from nodeloom.control import ControlRegistry
from nodeloom.transport import HttpTransport
from nodeloom.config import NodeLoomConfig


class ControlRegistryTests(unittest.TestCase):

    def test_get_unknown_agent_returns_default(self):
        registry = ControlRegistry()
        state = registry.get("agent-1")
        self.assertFalse(state.halted)
        self.assertEqual(state.halt_source, "none")
        self.assertEqual(state.revision, 0)
        self.assertEqual(state.require_guardrails, "OFF")

    def test_update_from_payload_marks_agent_halted(self):
        registry = ControlRegistry()
        registry.update_from_payload({
            "agent_name": "agent-1",
            "halted": True,
            "halt_source": "agent",
            "halt_reason": "policy violation",
            "revision": 5,
            "require_guardrails": "OFF",
            "guardrail_session_ttl_seconds": 300,
        })
        state = registry.get("agent-1")
        self.assertTrue(state.halted)
        self.assertEqual(state.halt_source, "agent")
        self.assertEqual(state.halt_reason, "policy violation")
        self.assertEqual(state.revision, 5)

    def test_team_halt_overrides_agent_state_for_all_agents(self):
        registry = ControlRegistry()
        registry.update_from_payload({
            "agent_name": "agent-known",
            "halted": False,
            "halt_source": "none",
            "revision": 1,
            "require_guardrails": "OFF",
        })
        # Now team-wide halt arrives.
        registry.update_from_payload({
            "agent_name": "agent-known",
            "halted": True,
            "halt_source": "team",
            "halt_reason": "incident response",
            "revision": 1_000_000,
            "require_guardrails": "OFF",
        })

        for name in ("agent-known", "never-seen-agent"):
            state = registry.get(name)
            self.assertTrue(state.halted, name)
            self.assertEqual(state.halt_source, "team", name)
            self.assertEqual(state.halt_reason, "incident response", name)

    def test_stale_revision_is_ignored(self):
        registry = ControlRegistry()
        registry.update_from_payload({
            "agent_name": "agent-1",
            "halted": True,
            "halt_source": "agent",
            "halt_reason": "current",
            "revision": 10,
            "require_guardrails": "OFF",
        })
        registry.update_from_payload({
            "agent_name": "agent-1",
            "halted": False,  # would clear halt if applied
            "halt_source": "none",
            "revision": 3,    # stale
            "require_guardrails": "OFF",
        })
        self.assertTrue(registry.get("agent-1").halted)

    def test_guardrail_session_round_trip(self):
        registry = ControlRegistry()
        now = time.monotonic()
        registry.record_guardrail_session("agent-1", "sess-abc", 300, now)
        self.assertEqual(registry.take_guardrail_session("agent-1", now + 1), "sess-abc")

    def test_expired_guardrail_session_returns_none(self):
        registry = ControlRegistry()
        now = time.monotonic()
        registry.record_guardrail_session("agent-1", "sess-abc", 5, now)
        self.assertIsNone(registry.take_guardrail_session("agent-1", now + 10))

    def test_missing_session_id_is_noop(self):
        registry = ControlRegistry()
        registry.record_guardrail_session("agent-1", "", 300, time.monotonic())
        self.assertIsNone(registry.take_guardrail_session("agent-1", time.monotonic()))


class TraceHaltTests(unittest.TestCase):

    def test_trace_raises_agent_halted_error(self):
        client = NodeLoom(api_key="sdk_test", control_poll_interval=0)
        try:
            # Inject halt directly via the registry
            client._control_registry.update_from_payload({
                "agent_name": "halted-agent",
                "halted": True,
                "halt_source": "agent",
                "halt_reason": "manual",
                "revision": 1,
                "require_guardrails": "OFF",
            })
            with self.assertRaises(AgentHaltedError) as ctx:
                client.trace("halted-agent")
            self.assertEqual(ctx.exception.agent_name, "halted-agent")
            self.assertEqual(ctx.exception.source, "agent")
            self.assertEqual(ctx.exception.reason, "manual")
        finally:
            client.shutdown(timeout=1.0)

    def test_team_halt_blocks_unknown_agents(self):
        client = NodeLoom(api_key="sdk_test", control_poll_interval=0)
        try:
            client._control_registry.update_from_payload({
                "agent_name": None,
                "halted": True,
                "halt_source": "team",
                "halt_reason": "incident",
                "revision": 99_999,
                "require_guardrails": "OFF",
            })
            with self.assertRaises(AgentHaltedError):
                client.trace("never-seen-agent")
        finally:
            client.shutdown(timeout=1.0)

    def test_trace_emits_guardrail_session_id(self):
        client = NodeLoom(api_key="sdk_test", control_poll_interval=0)
        try:
            client._control_registry.record_guardrail_session(
                "ok-agent", "sess-xyz", 300, time.monotonic()
            )
            trace = client.trace("ok-agent")
            try:
                # Drain the queue and find the trace_start event
                events = client._queue.drain(10)
                start = next(e for e in events if e["type"] == "trace_start")
                self.assertEqual(start["guardrail_session_id"], "sess-xyz")
            finally:
                trace.end()
        finally:
            client.shutdown(timeout=1.0)


class TransportControlPiggyBackTests(unittest.TestCase):

    def _config(self):
        return NodeLoomConfig(api_key="sdk_test", endpoint="https://api.example.com")

    @patch("nodeloom.transport.requests.Session")
    def test_response_control_updates_registry(self, mock_session_cls):
        registry = ControlRegistry()
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "accepted": 1,
            "rejected": 0,
            "errors": [],
            "control": {
                "agent_name": "agent-1",
                "halted": True,
                "halt_source": "agent",
                "halt_reason": "policy",
                "revision": 7,
                "require_guardrails": "HARD",
                "guardrail_session_ttl_seconds": 300,
            },
        }
        mock_session.post.return_value = mock_response

        transport = HttpTransport(self._config(), control_registry=registry)
        result = transport.send_batch([{"type": "span"}])

        self.assertIsNotNone(result)
        state = registry.get("agent-1")
        self.assertTrue(state.halted)
        self.assertEqual(state.require_guardrails, "HARD")

    @patch("nodeloom.transport.requests.Session")
    def test_response_without_control_field_is_safe(self, mock_session_cls):
        registry = ControlRegistry()
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"accepted": 1, "rejected": 0, "errors": []}
        mock_session.post.return_value = mock_response

        transport = HttpTransport(self._config(), control_registry=registry)
        transport.send_batch([{"type": "span"}])

        # No agents observed; default state for queries.
        state = registry.get("agent-1")
        self.assertFalse(state.halted)


class ApiClientGuardrailSessionTests(unittest.TestCase):

    @patch("nodeloom.api.requests.Session")
    def test_check_guardrails_caches_session_id(self, mock_session_cls):
        registry = ControlRegistry()
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_response = MagicMock(ok=True, status_code=200)
        mock_response.json.return_value = {
            "passed": True,
            "violations": [],
            "redactedContent": None,
            "checks": [],
            "guardrailSessionId": "sess-321",
        }
        mock_session.request.return_value = mock_response

        client = ApiClient(api_key="sdk_test", control_registry=registry)
        result = client.check_guardrails(team_id="team", text="ok", agent_name="agent-1")

        self.assertEqual(result["guardrailSessionId"], "sess-321")
        self.assertEqual(
            registry.take_guardrail_session("agent-1", time.monotonic()),
            "sess-321",
        )

    @patch("nodeloom.api.requests.Session")
    def test_get_agent_control_updates_registry(self, mock_session_cls):
        registry = ControlRegistry()
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_response = MagicMock(ok=True, status_code=200)
        mock_response.json.return_value = {
            "agent_name": "agent-1",
            "halted": True,
            "halt_source": "team",
            "halt_reason": "incident",
            "revision": 1_000_000,
            "require_guardrails": "OFF",
        }
        mock_session.request.return_value = mock_response

        client = ApiClient(api_key="sdk_test", control_registry=registry)
        client.get_agent_control("agent-1")

        # Team-wide halt is propagated to any known agent.
        self.assertTrue(registry.get("agent-1").halted)
        self.assertTrue(registry.get("any-other-agent").halted)


if __name__ == "__main__":
    unittest.main()
