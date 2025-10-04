"""Contract test for GET /collectors/status endpoint.

This test MUST fail initially (TDD approach).
"""

import pytest
from fastapi.testclient import TestClient


class TestCollectorStatus:
    """Contract tests for collector status endpoint."""

    def test_get_all_status(self, client: TestClient):
        """Test getting status of all collectors."""
        # Act
        response = client.get("/collectors/status")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "collectors" in data
        assert "total_active" in data
        assert "total_sdrs" in data
        assert isinstance(data["collectors"], list)

    def test_get_single_collector_status(self, client: TestClient):
        """Test getting status of single collector."""
        # Arrange
        session_id = "test-session-123"

        # Act
        response = client.get(f"/collectors/status/{session_id}")

        # Assert
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert data["session_id"] == session_id
            assert "status" in data
            assert "frequency_khz" in data
            assert "duration_seconds" in data
            assert "bytes_recorded" in data

    def test_status_with_filters(self, client: TestClient):
        """Test getting status with filters."""
        # Act
        response = client.get(
            "/collectors/status",
            params={"band": "20m", "status": "recording"},
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        collectors = data["collectors"]
        # All returned collectors should match filters
        for collector in collectors:
            assert collector.get("band") == "20m"
            assert collector["status"] == "recording"

    def test_status_health_check(self, client: TestClient):
        """Test health check status."""
        # Act
        response = client.get("/collectors/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "healthy" in data
        assert "redis_connected" in data
        assert "postgres_connected" in data
        assert "worker_count" in data

    def test_status_metrics(self, client: TestClient):
        """Test getting collector metrics."""
        # Act
        response = client.get("/collectors/metrics")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "total_hours_collected" in data
        assert "hours_today" in data
        assert "active_sdrs" in data
        assert "storage_used_gb" in data
        assert "collection_rate_per_hour" in data


@pytest.fixture
def client():
    """Create test client."""
    from src.api.main import app  # Will fail initially

    return TestClient(app)