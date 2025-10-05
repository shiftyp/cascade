"""Integration tests for Phase 3.4: Dashboard components."""

import os
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import json
from src.dashboard.web_dashboard import (
    app,
    get_collection_status,
    get_hourly_stats,
    get_sdr_performance,
    get_band_coverage,
    get_qa_samples
)


class TestDashboardIntegration:
    """Test dashboard integration components."""

    @pytest.fixture
    async def mock_pool(self):
        """Create mock database pool."""
        pool = AsyncMock()
        conn = AsyncMock()

        # Mock connection acquire
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock()

        return pool, conn

    @pytest.mark.asyncio
    async def test_get_collection_status(self, mock_pool):
        """Test fetching collection status."""
        pool, conn = mock_pool

        # Mock database response
        mock_row = {
            'total_sessions': 1500,
            'unique_sdrs': 25,
            'total_hours_collected': 12500.5,
            'avg_session_hours': 8.3,
            'total_storage_gb': 4567.8,
            'bands_covered': 6
        }
        conn.fetchrow.return_value = mock_row

        with patch('src.dashboard.web_dashboard.get_db_pool', return_value=pool):
            result = await get_collection_status()

        assert result == mock_row
        conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_hourly_stats(self, mock_pool):
        """Test fetching hourly statistics."""
        pool, conn = mock_pool

        # Mock database response
        mock_rows = [
            {'hour_bin': datetime.utcnow(), 'frequency_band': '20m',
             'sessions_started': 5, 'hours_collected': 3.5, 'active_sdrs': 4},
            {'hour_bin': datetime.utcnow() - timedelta(hours=1), 'frequency_band': '40m',
             'sessions_started': 3, 'hours_collected': 2.1, 'active_sdrs': 3}
        ]
        conn.fetch.return_value = mock_rows

        with patch('src.dashboard.web_dashboard.get_db_pool', return_value=pool):
            result = await get_hourly_stats(hours=24)

        assert len(result) == 2
        assert result[0]['frequency_band'] == '20m'
        conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_sdr_performance(self, mock_pool):
        """Test fetching SDR performance metrics."""
        pool, conn = mock_pool

        # Mock database response
        mock_rows = [
            {'sdr_id': 1, 'sdr_name': 'Test SDR 1', 'grid_square': 'FN42',
             'is_active': True, 'total_sessions': 100, 'total_hours': 850.0,
             'success_rate': 95.0, 'reliability_score': 0.95},
            {'sdr_id': 2, 'sdr_name': 'Test SDR 2', 'grid_square': 'IO91',
             'is_active': False, 'total_sessions': 50, 'total_hours': 400.0,
             'success_rate': 88.0, 'reliability_score': 0.88}
        ]
        conn.fetch.return_value = mock_rows

        with patch('src.dashboard.web_dashboard.get_db_pool', return_value=pool):
            result = await get_sdr_performance(active_only=False)

        assert len(result) == 2
        assert result[0]['sdr_name'] == 'Test SDR 1'
        assert result[0]['is_active'] is True

    @pytest.mark.asyncio
    async def test_get_band_coverage(self, mock_pool):
        """Test fetching band coverage statistics."""
        pool, conn = mock_pool

        # Mock database response
        mock_rows = [
            {'frequency_band': '20m', 'hour': 0, 'avg_hours': 5.2, 'avg_sdrs': 3},
            {'frequency_band': '20m', 'hour': 1, 'avg_hours': 4.8, 'avg_sdrs': 3},
            {'frequency_band': '40m', 'hour': 0, 'avg_hours': 6.1, 'avg_sdrs': 4},
            {'frequency_band': '40m', 'hour': 1, 'avg_hours': 5.9, 'avg_sdrs': 4}
        ]
        conn.fetch.return_value = mock_rows

        with patch('src.dashboard.web_dashboard.get_db_pool', return_value=pool):
            result = await get_band_coverage()

        assert '20m' in result
        assert '40m' in result
        assert len(result['20m']) == 24  # 24 hours
        assert result['20m'][0] == 5.2
        assert result['20m'][1] == 4.8

    @pytest.mark.asyncio
    async def test_get_qa_samples(self, mock_pool):
        """Test fetching QA samples."""
        pool, conn = mock_pool

        # Mock database response
        mock_rows = [
            {'id': 'sample1', 'recording_session_id': 'session1',
             'timestamp': datetime.utcnow(), 'frequency_band': '20m',
             'quality_score': 85.0, 'file_path': '/data/sample1.flac',
             'is_quarantined': False, 'sdr_name': 'SDR1', 'location_grid': 'FN42'},
            {'id': 'sample2', 'recording_session_id': 'session2',
             'timestamp': datetime.utcnow(), 'frequency_band': '40m',
             'quality_score': 92.0, 'file_path': '/data/sample2.flac',
             'is_quarantined': False, 'sdr_name': 'SDR2', 'location_grid': 'IO91'}
        ]
        conn.fetch.return_value = mock_rows

        with patch('src.dashboard.web_dashboard.get_db_pool', return_value=pool):
            result = await get_qa_samples(band='20m', min_quality=80.0, limit=10)

        assert len(result) == 2
        assert result[0]['id'] == 'sample1'
        assert result[0]['quality_score'] == 85.0

    @pytest.mark.asyncio
    async def test_dashboard_health_check(self, mock_pool):
        """Test dashboard health check endpoint."""
        pool, conn = mock_pool
        conn.fetchval.return_value = 1

        from src.dashboard.web_dashboard import health_check

        with patch('src.dashboard.web_dashboard.get_db_pool', return_value=pool):
            result = await health_check()

        assert result['status'] == 'healthy'
        assert result['database'] == 'connected'

    @pytest.mark.asyncio
    async def test_dashboard_websocket_updates(self):
        """Test WebSocket real-time updates."""
        # This test would require a more complex setup with WebSocket testing
        # For now, we'll test that the WebSocket endpoint exists
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Verify WebSocket endpoint is registered
        assert '/ws/updates' in [route.path for route in app.routes]

    def test_terminal_dashboard_data_fetch(self):
        """Test terminal dashboard data fetching."""
        from src.dashboard.terminal_dashboard import DashboardData

        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': 5432,
            'database': 'test',
            'user': 'test'
        }

        dashboard = DashboardData(db_config)

        # Test cache TTL
        assert dashboard.cache_ttl == 30

        # Test that queries are defined
        with patch.object(dashboard, '_fetch_query') as mock_fetch:
            mock_fetch.return_value = [{'test': 'data'}]

            # Test various data fetches
            result = dashboard.get_collection_status()
            assert result == {'test': 'data'}

            dashboard.get_hourly_stats(24)
            mock_fetch.assert_called()

            dashboard.get_sdr_performance()
            mock_fetch.assert_called()

    def test_sql_views_exist(self):
        """Test that SQL views are properly defined."""
        import os

        views_file = '/workspaces/cascade/modules/data/src/dashboard/views.sql'
        assert os.path.exists(views_file)

        with open(views_file, 'r') as f:
            sql_content = f.read()

        # Check that all expected views are defined
        expected_views = [
            'v_collection_status',
            'v_hourly_collection_stats',
            'v_sdr_performance',
            'v_propagation_summary',
            'v_qrn_coverage',
            'v_space_weather_events',
            'v_storage_usage',
            'v_qa_sample_quality',
            'v_correlation_completeness',
            'v_band_coverage'
        ]

        for view in expected_views:
            assert f'CREATE VIEW {view}' in sql_content, f"Missing view: {view}"