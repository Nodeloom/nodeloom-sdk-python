"""Tests for the REST API client."""

import pytest
from unittest.mock import MagicMock, patch
from nodeloom.api import ApiClient, ApiError


class TestApiClient:
    """Test ApiClient initialization and configuration."""

    def test_sets_authorization_header(self):
        client = ApiClient(api_key="sdk_test123")
        assert client._session.headers["Authorization"] == "Bearer sdk_test123"

    def test_sets_content_type_header(self):
        client = ApiClient(api_key="sdk_test123")
        assert client._session.headers["Content-Type"] == "application/json"

    def test_strips_trailing_slash_from_endpoint(self):
        client = ApiClient(api_key="sdk_test", endpoint="https://example.com/")
        assert client._endpoint == "https://example.com"

    def test_default_endpoint(self):
        client = ApiClient(api_key="sdk_test")
        assert client._endpoint == "https://api.nodeloom.io"

    def test_default_timeout_is_30_seconds(self):
        # Matches the Go/Java SDKs so behavior is consistent across languages.
        # Without an explicit timeout, requests.Session.request would block
        # forever on a hung backend.
        client = ApiClient(api_key="sdk_test")
        assert client._request_timeout_seconds == 30.0

    def test_timeout_is_forwarded_to_session(self):
        client = ApiClient(api_key="sdk_test", request_timeout_seconds=2.5)
        mock_response = MagicMock(ok=True, status_code=200)
        mock_response.json.return_value = {}
        with patch.object(client._session, "request", return_value=mock_response) as mock_req:
            client.request("GET", "/api/ping")
        assert mock_req.call_args.kwargs["timeout"] == 2.5


class TestApiRequest:
    """Test the generic request method."""

    def test_successful_get_request(self):
        client = ApiClient(api_key="sdk_test")
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": "wf-1", "name": "Test"}]

        with patch.object(client._session, "request", return_value=mock_response) as mock_req:
            result = client.request("GET", "/api/workflows", params={"teamId": "t1"})

        mock_req.assert_called_once_with(
            method="GET",
            url="https://api.nodeloom.io/api/workflows",
            json=None,
            params={"teamId": "t1"},
            timeout=30.0,
        )
        assert result == [{"id": "wf-1", "name": "Test"}]

    def test_successful_post_request(self):
        client = ApiClient(api_key="sdk_test")
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"executionId": "ex-1"}

        with patch.object(client._session, "request", return_value=mock_response):
            result = client.request("POST", "/api/workflows/wf-1/execute", body={"input": "test"})

        assert result == {"executionId": "ex-1"}

    def test_204_returns_none(self):
        client = ApiClient(api_key="sdk_test")
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 204

        with patch.object(client._session, "request", return_value=mock_response):
            result = client.request("DELETE", "/api/some/resource")

        assert result is None

    def test_error_response_raises_api_error(self):
        client = ApiClient(api_key="sdk_test")
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 403
        mock_response.json.return_value = {"error": "Access denied"}
        mock_response.text = "Access denied"

        with patch.object(client._session, "request", return_value=mock_response):
            with pytest.raises(ApiError) as exc_info:
                client.request("GET", "/api/workflows")

        assert exc_info.value.status_code == 403
        assert "Access denied" in str(exc_info.value)

    def test_error_response_with_non_json_body(self):
        client = ApiClient(api_key="sdk_test")
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("not json")
        mock_response.text = "Internal Server Error"

        with patch.object(client._session, "request", return_value=mock_response):
            with pytest.raises(ApiError) as exc_info:
                client.request("GET", "/api/workflows")

        assert exc_info.value.status_code == 500


class TestConvenienceMethods:
    """Test convenience methods."""

    def test_list_workflows(self):
        client = ApiClient(api_key="sdk_test")
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch.object(client._session, "request", return_value=mock_response) as mock_req:
            client.list_workflows("team-1")

        mock_req.assert_called_once_with(
            method="GET",
            url="https://api.nodeloom.io/api/workflows",
            json=None,
            params={"teamId": "team-1"},
            timeout=30.0,
        )

    def test_execute_workflow(self):
        client = ApiClient(api_key="sdk_test")
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "ex-1"}

        with patch.object(client._session, "request", return_value=mock_response) as mock_req:
            client.execute_workflow("wf-1", {"query": "hello"})

        mock_req.assert_called_once_with(
            method="POST",
            url="https://api.nodeloom.io/api/workflows/wf-1/execute",
            json={"query": "hello"},
            params=None,
            timeout=30.0,
        )

    def test_get_execution(self):
        client = ApiClient(api_key="sdk_test")
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "ex-1", "status": "completed"}

        with patch.object(client._session, "request", return_value=mock_response) as mock_req:
            client.get_execution("ex-1")

        mock_req.assert_called_once_with(
            method="GET",
            url="https://api.nodeloom.io/api/executions/ex-1",
            json=None,
            params=None,
            timeout=30.0,
        )


class TestApiClientFromNodeLoom:
    """Test API client access from main NodeLoom client."""

    def test_api_property_returns_api_client(self):
        from nodeloom import NodeLoom
        client = NodeLoom(api_key="sdk_test123", enabled=False)
        api = client.api
        assert isinstance(api, ApiClient)
        assert api._api_key == "sdk_test123"

    def test_api_property_is_cached(self):
        from nodeloom import NodeLoom
        client = NodeLoom(api_key="sdk_test123", enabled=False)
        api1 = client.api
        api2 = client.api
        assert api1 is api2
