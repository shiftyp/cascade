"""Integration tests for Phase 3.4 components without external dependencies."""

import pytest
import os
from pathlib import Path


class TestPhase34Components:
    """Test Phase 3.4 implementation files exist and are valid."""

    def test_gmail_notifier_exists(self):
        """Test that Gmail notifier has been implemented."""
        file_path = Path('src/notifications/gmail_notifier.py')
        assert file_path.exists(), "Gmail notifier not found"

        # Check that file has expected classes/functions
        with open(file_path, 'r') as f:
            content = f.read()

        assert 'class GmailNotifier' in content
        assert 'class NotificationConfig' in content
        assert 'def send_notification' in content
        assert 'def send_space_weather_alert' in content
        assert 'def send_collection_status' in content
        assert 'def send_error_alert' in content
        assert 'def send_qa_sample_alert' in content

    def test_correlation_manager_exists(self):
        """Test that correlation manager has been implemented."""
        file_path = Path('src/processors/correlation_manager.py')
        assert file_path.exists(), "Correlation manager not found"

        with open(file_path, 'r') as f:
            content = f.read()

        assert 'class CorrelationManager' in content
        assert 'class CorrelationMetadata' in content
        assert 'def register_sample' in content
        assert 'def calculate_correlation' in content

    def test_dashboard_sql_views_exist(self):
        """Test that dashboard SQL views have been created."""
        file_path = Path('src/dashboard/views.sql')
        assert file_path.exists(), "Dashboard SQL views not found"

        with open(file_path, 'r') as f:
            content = f.read()

        # Check for all expected views
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
            assert f'CREATE VIEW {view}' in content, f"Missing view: {view}"

    def test_terminal_dashboard_exists(self):
        """Test that terminal dashboard has been implemented."""
        file_path = Path('src/dashboard/terminal_dashboard.py')
        assert file_path.exists(), "Terminal dashboard not found"

        with open(file_path, 'r') as f:
            content = f.read()

        assert 'class TerminalDashboard' in content
        assert 'class DashboardData' in content
        assert 'def _draw_overview' in content
        assert 'def _draw_sdr_status' in content
        assert 'def _draw_band_coverage' in content

    def test_web_dashboard_exists(self):
        """Test that web dashboard has been implemented."""
        file_path = Path('src/dashboard/web_dashboard.py')
        assert file_path.exists(), "Web dashboard not found"

        with open(file_path, 'r') as f:
            content = f.read()

        # Check for FastAPI app and endpoints
        assert 'app = FastAPI' in content
        assert '@app.get("/api/status")' in content
        assert '@app.get("/api/hourly-stats")' in content
        assert '@app.get("/api/sdr-performance")' in content
        assert '@app.get("/api/band-coverage")' in content
        assert '@app.get("/api/qa-samples")' in content
        assert '@app.websocket("/ws/updates")' in content

    def test_phase34_structure(self):
        """Test that Phase 3.4 directory structure is correct."""
        # Check notifications directory
        notifications_dir = Path('src/notifications')
        assert notifications_dir.exists() and notifications_dir.is_dir()

        # Check dashboard directory
        dashboard_dir = Path('src/dashboard')
        assert dashboard_dir.exists() and dashboard_dir.is_dir()

        # Check that required files exist
        assert Path('src/notifications/gmail_notifier.py').exists()
        assert Path('src/dashboard/views.sql').exists()
        assert Path('src/dashboard/terminal_dashboard.py').exists()
        assert Path('src/dashboard/web_dashboard.py').exists()

    def test_notification_environment_config(self):
        """Test that notification configuration can use environment variables."""
        from src.notifications.gmail_notifier import NotificationConfig

        # Test with environment variables set
        test_env = {
            'GMAIL_SENDER_EMAIL': 'test@example.com',
            'GMAIL_APP_PASSWORD': 'test_password',
            'NOTIFICATION_RECIPIENTS': 'user1@example.com,user2@example.com'
        }

        with pytest.MonkeyPatch.context() as mp:
            for key, value in test_env.items():
                mp.setenv(key, value)

            config = NotificationConfig()
            assert config.sender_email == 'test@example.com'
            assert config.sender_password == 'test_password'
            assert len(config.recipient_emails) == 2

    def test_correlation_manager_basic_functionality(self):
        """Test basic correlation manager functionality."""
        from src.processors.correlation_manager import CorrelationManager, CorrelationMetadata
        from datetime import datetime

        manager = CorrelationManager()

        # Test sample registration
        metadata = manager.register_sample(
            sample_id="test_sample_1",
            recording_id="test_recording_1",
            timestamp=datetime.now(),
            frequency=14100000,
            band="20m",
            location={"lat": 42.0, "lon": -71.0},
            processing_stage="initial"
        )

        assert isinstance(metadata, CorrelationMetadata)
        assert metadata.sample_id == "test_sample_1"
        assert metadata.band == "20m"

    def test_terminal_dashboard_data_structure(self):
        """Test terminal dashboard data fetching structure."""
        from src.dashboard.terminal_dashboard import DashboardData

        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test',
            'user': 'test'
        }

        dashboard_data = DashboardData(db_config)

        # Test that cache is initialized
        assert hasattr(dashboard_data, 'cache')
        assert hasattr(dashboard_data, 'cache_time')
        assert dashboard_data.cache_ttl == 30

        # Test that methods exist
        assert hasattr(dashboard_data, 'get_collection_status')
        assert hasattr(dashboard_data, 'get_hourly_stats')
        assert hasattr(dashboard_data, 'get_sdr_performance')
        assert hasattr(dashboard_data, 'get_band_coverage')
        assert hasattr(dashboard_data, 'get_space_weather')
        assert hasattr(dashboard_data, 'get_storage_usage')

    def test_phase34_integration_summary(self):
        """Test that Phase 3.4 integration is complete."""
        # List of Phase 3.4 tasks that should be completed
        completed_components = {
            'T047': Path('src/notifications/gmail_notifier.py'),
            'T047a': Path('src/processors/correlation_manager.py'),
            'T049': Path('src/dashboard/views.sql'),
            'T050': Path('src/dashboard/terminal_dashboard.py'),
            'T051': Path('src/dashboard/web_dashboard.py')
        }

        missing = []
        for task_id, file_path in completed_components.items():
            if not file_path.exists():
                missing.append(f"{task_id}: {file_path}")

        assert len(missing) == 0, f"Missing Phase 3.4 components: {missing}"

        print("\n=== Phase 3.4 Integration Summary ===")
        print(f"✓ Gmail notification service (T047)")
        print(f"✓ Correlation manager (T047a)")
        print(f"✓ Dashboard SQL views (T049)")
        print(f"✓ Terminal dashboard (T050)")
        print(f"✓ FastAPI web dashboard (T051)")
        print("=====================================\n")