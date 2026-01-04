import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


class TestWebServer:
    @pytest.fixture
    def client(self):
        with patch("src.web.server.settings"):
            from src.web.server import app
            return TestClient(app)

    def test_root_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Agents Arguing" in response.text
        assert "text/html" in response.headers["content-type"]

    def test_docs_endpoint_exists(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_endpoint_exists(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Agents Arguing"


class TestDebateAPI:
    @pytest.fixture
    def client(self):
        with patch("src.web.server.settings"):
            from src.web.server import app
            return TestClient(app)

    def test_start_debate_creates_session(self, client):
        with patch("src.web.server.StreamingDebateSession") as mock_session:
            mock_instance = MagicMock()
            mock_instance.initialize = MagicMock(return_value=None)
            mock_session.return_value = mock_instance

            response = client.post(
                "/api/debate/start",
                json={
                    "topic": "Test topic",
                    "pro_name": "Alice",
                    "con_name": "Bob",
                    "num_rounds": 2,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert data["topic"] == "Test topic"

    def test_get_status_returns_404_for_unknown_session(self, client):
        response = client.get("/api/debate/unknown-id/status")
        assert response.status_code == 404

    def test_stop_debate_returns_404_for_unknown_session(self, client):
        response = client.post("/api/debate/unknown-id/stop")
        assert response.status_code == 404

    def test_delete_session_returns_404_for_unknown_session(self, client):
        response = client.delete("/api/debate/unknown-id")
        assert response.status_code == 404
