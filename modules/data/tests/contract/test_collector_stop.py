"""Contract test for POST /collectors/stop endpoint.

This test MUST fail initially (TDD approach).
"""

import pytest
from fastapi.testclient import TestClient


class TestCollectorStop:
    """Contract tests for stopping collection."""

    def test_stop_active_recording(self, client: TestClient):
        """Test stopping an active recording."""
        # Arrange
        session_id = "active-session-123"

        # Act
        response = client.post(f"/collectors/stop", json={"session_id": session_id})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
        assert data["session_id"] == session_id
        assert "stop_time" in data
        assert "bytes_recorded" in data

    def test_stop_nonexistent_session(self, client: TestClient):
        """Test stopping non-existent session."""
        # Arrange
        session_id = "nonexistent-123"

        # Act
        response = client.post(f"/collectors/stop", json={"session_id": session_id})

        # Assert
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["error"].lower()

    def test_stop_already_stopped(self, client: TestClient):
        """Test stopping already stopped session."""
        # Arrange
        session_id = "already-stopped-123"

        # Act - Stop twice
        response1 = client.post(f"/collectors/stop", json={"session_id": session_id})
        response2 = client.post(f"/collectors/stop", json={"session_id": session_id})

        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 409
        data2 = response2.json()
        assert "already stopped" in data2["error"].lower()

    def test_stop_with_save_options(self, client: TestClient):
        """Test stopping with save options."""
        # Arrange
        payload = {
            "session_id": "save-options-123",
            "save_to_tigris": True,
            "compress": True,
            "delete_local": False,
        }

        # Act
        response = client.post(f"/collectors/stop", json=payload)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["saved_to_tigris"] == True
        assert data["compressed"] == True
        assert "file_path" in data

    def test_force_stop(self, client: TestClient):
        """Test force stopping a stuck session."""
        # Arrange
        payload = {
            "session_id": "stuck-session-123",
            "force": True,
        }

        # Act
        response = client.post(f"/collectors/stop", json=payload)

        # Assert
        assert response.status_code in [200, 202]  # OK or Accepted
        data = response.json()
        assert data["status"] in ["stopped", "force_stopped"]


@pytest.fixture
def client():
    """Create test client."""
    from src.api.main import app  # Will fail initially

    return TestClient(app)