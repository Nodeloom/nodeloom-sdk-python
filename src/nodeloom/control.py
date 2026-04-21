"""Remote-control state shared between transport, polling, and the trace path.

The SDK keeps a small thread-safe registry of agent control state so that:

  * The telemetry transport can update it from every batch response.
  * An optional poller thread can refresh it for sparse-traffic agents.
  * The check_guardrails caller can stash a short-lived ``guardrail_session_id``
    that the trace path will attach to the next ``trace_start`` event for HARD
    required-guardrail enforcement.
  * The ``trace()`` factory can raise :class:`AgentHaltedError` if the backend
    has halted the agent or the team it belongs to.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("nodeloom.control")


class AgentHaltedError(RuntimeError):
    """Raised when an SDK operation targets an agent the backend has halted.

    Carries the resolved control payload so callers can introspect the halt
    reason and source ("agent" or "team") for logging and incident response.
    """

    def __init__(self, agent_name: str, reason: Optional[str], source: str,
                 revision: int, payload: Dict[str, Any]) -> None:
        self.agent_name = agent_name
        self.reason = reason
        self.source = source
        self.revision = revision
        self.payload = payload
        message = (
            f"Agent '{agent_name}' is halted (source={source}, revision={revision})"
        )
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message)


@dataclass
class AgentControlState:
    """Per-agent control snapshot.

    ``revision`` advances monotonically in the backend (composite of team
    and per-agent revision). The state is only mutated when a fresher
    revision arrives, which makes the registry update idempotent.
    """

    halted: bool = False
    halt_reason: Optional[str] = None
    halt_source: str = "none"
    revision: int = 0
    require_guardrails: str = "OFF"
    guardrail_session_ttl_seconds: int = 300
    guardrail_session_id: Optional[str] = None
    # Monotonic time (time.monotonic()) at which the cached session id expires.
    guardrail_session_expires_at: float = 0.0
    raw_payload: Dict[str, Any] = field(default_factory=dict)


class ControlRegistry:
    """Thread-safe registry of :class:`AgentControlState` per agent name.

    Per-agent state and a team-wide override are tracked independently. When
    the team is halted, every read returns ``halted=True`` regardless of the
    individual agent's state.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._agents: Dict[str, AgentControlState] = {}
        self._team_halted: bool = False
        self._team_halt_reason: Optional[str] = None
        self._team_revision: int = 0

    # -- Reads -------------------------------------------------------------

    def get(self, agent_name: str) -> AgentControlState:
        """Return a *snapshot* of the state for an agent.

        Always returns a fresh dataclass so callers can read fields without
        holding the lock. Unknown agents inherit any active team-wide halt.
        """
        with self._lock:
            base = self._agents.get(agent_name)
            if base is None:
                state = AgentControlState()
            else:
                state = AgentControlState(
                    halted=base.halted,
                    halt_reason=base.halt_reason,
                    halt_source=base.halt_source,
                    revision=base.revision,
                    require_guardrails=base.require_guardrails,
                    guardrail_session_ttl_seconds=base.guardrail_session_ttl_seconds,
                    guardrail_session_id=base.guardrail_session_id,
                    guardrail_session_expires_at=base.guardrail_session_expires_at,
                    raw_payload=dict(base.raw_payload),
                )

            if self._team_halted:
                state.halted = True
                state.halt_source = "team"
                state.halt_reason = self._team_halt_reason
            return state

    def is_halted(self, agent_name: str) -> bool:
        with self._lock:
            if self._team_halted:
                return True
            base = self._agents.get(agent_name)
            return bool(base and base.halted)

    def known_agents(self) -> list:
        with self._lock:
            return list(self._agents.keys())

    # -- Writes ------------------------------------------------------------

    def update_from_payload(self, payload: Optional[Dict[str, Any]]) -> None:
        """Merge a control payload returned by the backend into the registry.

        Stale revisions are ignored to avoid overwriting newer poll results
        with an older piggy-backed copy that arrived out of order.
        """
        if not payload:
            return

        agent_name = payload.get("agent_name")
        revision = int(payload.get("revision", 0) or 0)
        halted = bool(payload.get("halted"))
        halt_source = payload.get("halt_source") or "none"
        halt_reason = payload.get("halt_reason")
        require_guardrails = (payload.get("require_guardrails") or "OFF").upper()
        # Clamp TTL to a sane range: the backend default is 300s; anything
        # outside [1, 86_400] is treated as garbage and replaced with the
        # default. Protects against a buggy/malicious server.
        raw_ttl = int(payload.get("guardrail_session_ttl_seconds", 300) or 300)
        ttl = raw_ttl if 1 <= raw_ttl <= 86_400 else 300

        with self._lock:
            # Team-wide flag derived from halt_source. Agent-source payloads never
            # mutate team state: only team-source payloads can set or clear the
            # team-wide halt, and only when their revision is fresher. This
            # prevents a piggy-backed agent response arriving late from clobbering
            # a team halt that was issued after it.
            if halt_source == "team" and revision >= self._team_revision:
                self._team_halted = halted
                self._team_halt_reason = halt_reason
                self._team_revision = revision

            if not agent_name:
                return

            existing = self._agents.get(agent_name)
            if existing is None:
                existing = AgentControlState()
                self._agents[agent_name] = existing

            if revision < existing.revision:
                logger.debug(
                    "Ignoring stale control payload for %s (rev %d < cached %d)",
                    agent_name, revision, existing.revision,
                )
                return

            existing.halted = halted and halt_source != "team"  # team flag handled above
            existing.halt_source = halt_source
            existing.halt_reason = halt_reason if halt_source != "team" else None
            existing.revision = revision
            existing.require_guardrails = require_guardrails
            existing.guardrail_session_ttl_seconds = ttl
            existing.raw_payload = dict(payload)

    # -- Guardrail session id ---------------------------------------------

    def record_guardrail_session(self, agent_name: Optional[str], session_id: str,
                                 ttl_seconds: int, monotonic_now: float) -> None:
        """Stash a guardrail session id returned by check_guardrails.

        ``monotonic_now`` is parameterised so tests can use a fake clock.
        """
        if not session_id or not agent_name:
            return
        with self._lock:
            state = self._agents.get(agent_name)
            if state is None:
                state = AgentControlState()
                self._agents[agent_name] = state
            state.guardrail_session_id = session_id
            state.guardrail_session_expires_at = monotonic_now + max(1, ttl_seconds)

    def take_guardrail_session(self, agent_name: str, monotonic_now: float) -> Optional[str]:
        """Return a guardrail session id that is still valid, or ``None``.

        Does *not* clear the cached id: the same session can correlate multiple
        traces within its TTL window. Backend HARD-mode enforcement does the
        binding revocation when traces actually arrive.
        """
        with self._lock:
            state = self._agents.get(agent_name)
            if state is None or not state.guardrail_session_id:
                return None
            if monotonic_now >= state.guardrail_session_expires_at:
                state.guardrail_session_id = None
                state.guardrail_session_expires_at = 0.0
                return None
            return state.guardrail_session_id


def raise_if_halted(registry: ControlRegistry, agent_name: str) -> None:
    """Raise :class:`AgentHaltedError` if the registry says the agent is halted."""
    state = registry.get(agent_name)
    if state.halted:
        raise AgentHaltedError(
            agent_name=agent_name,
            reason=state.halt_reason,
            source=state.halt_source,
            revision=state.revision,
            payload=state.raw_payload,
        )
