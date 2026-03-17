"""REST API client for NodeLoom.

SDK tokens can authenticate against all NodeLoom API endpoints.
This module provides a typed client for common operations.
"""

import requests
from typing import Any, Dict, List, Optional


class ApiClient:
    """HTTP client for the NodeLoom REST API.

    Uses the same SDK token and endpoint as the telemetry client.
    """

    def __init__(self, api_key: str, endpoint: str = "https://api.nodeloom.io") -> None:
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
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
        body.update(kwargs)
        return self.request("POST", f"/api/guardrails/check", body=body, params={"teamId": team_id})

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
