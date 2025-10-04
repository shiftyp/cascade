"""Contract test for POST /collectors/start endpoint.

This test MUST fail initially (TDD approach).
"""

import pytest
from fastapi.testclient import TestClient


class TestCollectorStart:
    """Contract tests for starting collection."""

    def test_start_recording_success(self, client: TestClient):
        """Test starting a recording session."""
        # Arrange
        payload = {
            "session_id": "test-session-123",
            "duration_seconds": 300,
            "output_format": "flac",
            "metadata": {
                "band": "20m",
                "kiwisdr_source": "websdr.ewi.utwente.nl:8073",
            },
        }

        # Act
        response = client.post("/collectors/start", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recording"
        assert data["session_id"] == payload["session_id"]
        assert "start_time" in data
        assert data["duration_seconds"] == payload["duration_seconds"]

    def test_start_without_connection(self, client: TestClient):
        """Test starting recording without active connection."""
        # Arrange
        payload = {
            "session_id": "no-connection-123",
            "duration_seconds": 300,
        }

        # Act
        response = client.post("/collectors/start", json=payload)

        # Assert
        assert response.status_code == 409  # Conflict
        data = response.json()
        assert "not connected" in data["error"].lower()

    def test_start_duplicate_session(self, client: TestClient):
        """Test starting with duplicate session ID."""
        # Arrange
        payload = {
            "session_id": "duplicate-123",
            "duration_seconds": 300,
        }

        # Act - Start first session
        response1 = client.post("/collectors/start", json=payload)
        # Try to start another with same ID
        response2 = client.post("/collectors/start", json=payload)

        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 409
        data2 = response2.json()
        assert "already exists" in data2["error"].lower()

    def test_start_with_invalid_duration(self, client: TestClient):
        """Test starting with invalid duration."""
        # Arrange
        payload = {
            "session_id": "test-123",
            "duration_seconds": -10,  # Invalid
        }

        # Act
        response = client.post("/collectors/start", json=payload)

        # Assert
        assert response.status_code == 422
        data = response.json()
        assert "duration" in str(data["detail"]).lower()

    def test_start_with_schedule(self, client: TestClient):
        """Test starting with scheduled time."""
        # Arrange
        payload = {
            "session_id": "scheduled-123",
            "duration_seconds": 300,
            "start_at": "2024-01-01T12:00:00Z",
        }

        # Act
        response = client.post("/collectors/start", json=payload)

        # Assert
        assert response.status_code == 202  # Accepted
        data = response.json()
        assert data["status"] == "scheduled"
        assert data["start_at"] == payload["start_at"]


@pytest.fixture
def client():
    """Create test client."""
    from src.api.main import app  # Will fail initially

    return TestClient(app)