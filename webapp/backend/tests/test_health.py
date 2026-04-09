from fastapi.testclient import TestClient
from app.main import app


def test_health_returns_200():
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/health")
    assert response.status_code == 200


def test_health_schema(client):
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "version" in data
    assert data["status"] in ("healthy", "degraded")
    assert isinstance(data["models_loaded"], list)