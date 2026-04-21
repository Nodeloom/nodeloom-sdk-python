"""REST API client for NodeLoom.

SDK tokens can authenticate against all NodeLoom API endpoints.
This module provides a typed client for common operations.
"""

import time
import requests
from typing import Any, Dict, List, Optional

from nodeloom.control import ControlRegistry


class ApiClient:
    """HTTP client for the NodeLoom REST API.

    Uses the same SDK token and endpoint as the telemetry client.

    When a :class:`ControlRegistry` is supplied, ``check_guardrails`` updates
    it with any returned ``guardrailSessionId`` so subsequent traces can
    attach the id automatically (Phase 2 required-guardrail enforcement).
    """

    def __init__(self, api_key: str, endpoint: str = "https://api.nodeloom.io",
                 control_registry: Optional[ControlRegistry] = None) -> None:
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._control_registry = control_registry
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })

    def request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make an authenticated API request.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            path: API path (e.g., "/api/workflows")
            body: Optional JSON request body
            params: Optional query parameters

        Returns:
            Parsed JSON response

        Raises:
            ApiError: If the request fails with a non-2xx status code
        """
        url = f"{self._endpoint}{path}"
        response = self._session.request(
            method=method,
            url=url,
            json=body,
            params=params,
        )
        if not response.ok:
            error_body = None
            try:
                error_body = response.json()
            except Exception:
                pass
            raise ApiError(
                status_code=response.status_code,
                message=error_body.get("error", response.text) if error_body else response.text,
                response=error_body,
            )
        if response.status_code == 204:
            return None
        return response.json()

    # ── Workflow Operations ──────────────────────────────────────

    def list_workflows(self, team_id: str) -> List[Dict[str, Any]]:
        """List all workflows for a team."""
        return self.request("GET", "/api/workflows", params={"teamId": team_id})

    def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Get a workflow by ID."""
        return self.request("GET", f"/api/workflows/{workflow_id}")

    def execute_workflow(
        self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a workflow."""
        return self.request("POST", f"/api/workflows/{workflow_id}/execute", body=input_data or {})

    # ── Execution Operations ─────────────────────────────────────

    def list_executions(
        self, team_id: str, page: int = 0, size: int = 20
    ) -> Dict[str, Any]:
        """List executions for a team."""
        return self.request(
            "GET", "/api/executions", params={"teamId": team_id, "page": page, "size": size}
        )

    def get_execution(self, execution_id: str) -> Dict[str, Any]:
        """Get an execution by ID."""
        return self.request("GET", f"/api/executions/{execution_id}")

    # ── Credential Operations ────────────────────────────────────

    def list_credentials(self, team_id: str) -> List[Dict[str, Any]]:
        """List credentials for a team."""
        return self.request("GET", "/api/credentials", params={"teamId": team_id})

    # ── Guardrail Operations ────────────────────────────────────

    def check_guardrails(
        self,
        team_id: str,
        text: str,
        detect_prompt_injection: bool = False,
        redact_pii: bool = False,
        filter_content: bool = False,
        apply_custom_rules: bool = False,
        detect_semantic_manipulation: bool = False,
        on_violation: str = "BLOCKED",
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run guardrail checks on text content.

        Args:
            team_id: Team ID
            text: Text content to check
            detect_prompt_injection: Check for prompt injection attacks
            redact_pii: Detect and redact PII (emails, SSNs, etc.)
            filter_content: Filter harmful content
            apply_custom_rules: Apply team's custom guardrail rules
            detect_semantic_manipulation: Check semantic similarity against reference embeddings
            on_violation: Action on violation - "BLOCKED", "WARNED", or "LOGGED"
            agent_name: SDK agent name (enables incident playbook dispatch on violations)
            **kwargs: Additional config options (injectionSensitivity, piiTypes, etc.)

        Returns:
            Dict with keys: passed (bool), violations (list), redactedContent (str), checks (list)
        """
        body: Dict[str, Any] = {"text": text, "onViolation": on_violation}
        if detect_prompt_injection:
            body["detectPromptInjection"] = True
        if redact_pii:
            body["redactPii"] = True
        if filter_content:
            body["filterContent"] = True
        if apply_custom_rules:
            body["applyCustomRules"] = True
        if detect_semantic_manipulation:
            body["detectSemanticManipulation"] = True
        if agent_name:
            body["agentName"] = agent_name
        body.update(kwargs)
        response = self.request("POST", f"/api/guardrails/check", body=body, params={"teamId": team_id})

        # Cache the guardrail session id (when present) so the next trace_start
        # can attach it for HARD-mode required-guardrail enforcement.
        if (
            self._control_registry is not None
            and isinstance(response, dict)
            and agent_name
        ):
            session_id = response.get("guardrailSessionId")
            if session_id:
                # The backend's TTL is also returned via control payloads; default
                # to 300s when we have no fresher value in the registry.
                state = self._control_registry.get(agent_name)
                ttl = state.guardrail_session_ttl_seconds or 300
                self._control_registry.record_guardrail_session(
                    agent_name, session_id, ttl, time.monotonic()
                )
        return response

    # ── Feedback Operations ────────────────────────────────────

    def submit_feedback(
        self,
        execution_id: str,
        rating: int,
        comment: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        user_identifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit feedback for an execution."""
        body: Dict[str, Any] = {"execution_id": execution_id, "rating": rating}
        if comment:
            body["comment"] = comment
        if tags:
            body["tags"] = tags
        if trace_id:
            body["trace_id"] = trace_id
        if span_id:
            body["span_id"] = span_id
        if user_identifier:
            body["user_identifier"] = user_identifier
        return self.request("POST", "/api/sdk/v1/feedback", body=body)

    def list_feedback(
        self, execution_id: Optional[str] = None, page: int = 0, size: int = 20
    ) -> Dict[str, Any]:
        """List feedback for the team."""
        params: Dict[str, Any] = {"page": page, "size": size}
        if execution_id:
            params["execution_id"] = execution_id
        return self.request("GET", "/api/sdk/v1/feedback", params=params)

    # ── Sentiment Operations ─────────────────────────────────

    def analyze_sentiment(
        self, text: str, trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze text sentiment. Returns sentiment, score, confidence, emotions."""
        body: Dict[str, Any] = {"text": text}
        if trace_id:
            body["trace_id"] = trace_id
        return self.request("POST", "/api/sdk/v1/sentiment", body=body)

    # ── Cost Operations ──────────────────────────────────────

    def get_costs(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get cost summary for the team."""
        params: Dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if workflow_id:
            params["workflow_id"] = workflow_id
        return self.request("GET", "/api/sdk/v1/costs", params=params)

    # ── Webhook Operations ───────────────────────────────────

    def register_webhook(
        self,
        url: str,
        secret: Optional[str] = None,
        event_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Register an alert webhook."""
        body: Dict[str, Any] = {"url": url}
        if secret:
            body["secret"] = secret
        if event_types:
            body["event_types"] = event_types
        return self.request("POST", "/api/sdk/v1/alerts/webhooks", body=body)

    def list_webhooks(self) -> List[Dict[str, Any]]:
        """List registered alert webhooks."""
        return self.request("GET", "/api/sdk/v1/alerts/webhooks")

    def delete_webhook(self, webhook_id: str) -> None:
        """Delete an alert webhook."""
        self.request("DELETE", f"/api/sdk/v1/alerts/webhooks/{webhook_id}")

    # ── Prompt Operations ────────────────────────────────────

    def create_prompt(
        self,
        name: str,
        content: str,
        description: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        model_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or update a prompt template with a new version."""
        body: Dict[str, Any] = {"name": name, "content": content}
        if description:
            body["description"] = description
        if variables:
            body["variables"] = variables
        if model_hint:
            body["model_hint"] = model_hint
        return self.request("POST", "/api/sdk/v1/prompts", body=body)

    def get_prompt(
        self, name: str, version: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get a prompt template. Returns latest version if version not specified."""
        params = {"version": version} if version else None
        return self.request("GET", f"/api/sdk/v1/prompts/{name}", params=params)

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all prompt templates for the team."""
        return self.request("GET", "/api/sdk/v1/prompts")

    # ── Red Team Operations ──────────────────────────────────

    def start_red_team_scan(
        self, workflow_id: str, categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Start a red team scan against a workflow/agent."""
        body: Dict[str, Any] = {"workflow_id": workflow_id}
        if categories:
            body["categories"] = categories
        return self.request("POST", "/api/sdk/v1/redteam/scan", body=body)

    def get_red_team_scan(self, scan_id: str) -> Dict[str, Any]:
        """Get red team scan status and results."""
        return self.request("GET", f"/api/sdk/v1/redteam/scan/{scan_id}")

    # ── Evaluation Operations ────────────────────────────────

    def trigger_evaluation(self, execution_id: str) -> Dict[str, Any]:
        """Trigger LLM-as-Judge evaluation for an execution."""
        return self.request("POST", "/api/sdk/v1/evaluate", body={"execution_id": execution_id})

    # ── Metrics Operations ───────────────────────────────────

    def get_metrics(
        self,
        name: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get custom metrics aggregation."""
        params: Dict[str, Any] = {}
        if name:
            params["name"] = name
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self.request("GET", "/api/sdk/v1/metrics", params=params)

    # ── Agent Callback Operations ──────────────────────────────

    def set_callback_url(self, agent_name: str, callback_url: str) -> Dict[str, Any]:
        """Register a callback URL for red team testing of an SDK agent."""
        return self.request("POST", f"/api/sdk/v1/agents/{agent_name}/callback",
                           body={"callback_url": callback_url})

    def remove_callback_url(self, agent_name: str) -> None:
        """Remove the callback URL for an SDK agent."""
        self.request("DELETE", f"/api/sdk/v1/agents/{agent_name}/callback")

    def get_guardrail_config(self, agent_name: str) -> Dict[str, Any]:
        """Get the current guardrail configuration for an SDK agent (read-only). Configure via NodeLoom UI."""
        return self.request("GET", f"/api/sdk/v1/agents/{agent_name}/guardrails")

    # ── Remote Control (kill switch) ──────────────────────────────

    def get_agent_control(self, agent_name: str) -> Dict[str, Any]:
        """Fetch the current remote-control payload for an agent.

        Returns ``{halted, halt_reason, halt_source, revision,
        require_guardrails, guardrail_session_ttl_seconds, ...}``.

        When a :class:`ControlRegistry` is configured on this client, the
        registry is updated with the response so subsequent trace operations
        immediately observe the latest state.
        """
        response = self.request("GET", f"/api/sdk/v1/agents/{agent_name}/control")
        if self._control_registry is not None and isinstance(response, dict):
            self._control_registry.update_from_payload(response)
        return response

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()


class ApiError(Exception):
    """Raised when an API request returns a non-2xx status code."""

    def __init__(self, status_code: int, message: str, response: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"API error {status_code}: {message}")
