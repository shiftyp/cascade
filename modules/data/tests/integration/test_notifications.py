"""Integration tests for Phase 3.4: Gmail notification service."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import smtplib

from src.notifications.gmail_notifier import (
    GmailNotifier,
    NotificationConfig
)


class TestGmailNotificationIntegration:
    """Test Gmail notification service integration."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return NotificationConfig(
            sender_email="test@example.com",
            sender_password="test_password",
            recipient_emails=["recipient@example.com"]
        )

    @pytest.fixture
    def notifier(self, config):
        """Create notifier instance."""
        return GmailNotifier(config)

    def test_notification_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            'GMAIL_SENDER_EMAIL': 'env_sender@example.com',
            'GMAIL_APP_PASSWORD': 'env_password',
            'NOTIFICATION_RECIPIENTS': 'user1@example.com,user2@example.com'
        }):
            config = NotificationConfig()
            assert config.sender_email == "env_sender@example.com"
            assert config.sender_password == "env_password"
            assert config.recipient_emails == ["user1@example.com", "user2@example.com"]

    @patch('smtplib.SMTP')
    def test_send_notification_success(self, mock_smtp, notifier):
        """Test successful notification sending."""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        result = notifier.send_notification(
            subject="Test Alert",
            body="Test message body",
            priority="high"
        )

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.sendmail.assert_called_once()

    @patch('smtplib.SMTP')
    def test_space_weather_alert(self, mock_smtp, notifier):
        """Test space weather alert notification."""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        event_data = {
            "time": datetime.utcnow().isoformat(),
            "xray_class": "M5.0",
            "xray_flux": 5e-5,
            "active_collectors": 6,
            "target_collectors": 20
        }

        result = notifier.send_space_weather_alert(
            event_type="Solar Flare",
            event_data=event_data
        )

        assert result is True
        mock_server.sendmail.assert_called_once()

        # Check that high priority was set for M-class flare
        call_args = mock_server.sendmail.call_args
        message_str = call_args[0][2]
        assert "X-Priority: 1" in message_str or "Importance: high" in message_str

    @patch('smtplib.SMTP')
    def test_collection_status_report(self, mock_smtp, notifier):
        """Test daily collection status report."""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        result = notifier.send_collection_status(
            daily_hours=324.5,
            total_hours=15000.0,
            active_sdrs=8,
            failed_sdrs=["sdr1.example.com", "sdr2.example.com"],
            storage_used_gb=5432.1
        )

        assert result is True
        mock_server.sendmail.assert_called_once()

    @patch('smtplib.SMTP')
    def test_error_alert_notification(self, mock_smtp, notifier):
        """Test error alert notification."""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        result = notifier.send_error_alert(
            error_type="Connection Failure",
            error_message="Failed to connect to KiwiSDR at sdr.example.com",
            affected_components=["collector-1", "recorder-3"],
            suggested_action="Check network connectivity and SDR status"
        )

        assert result is True
        mock_server.sendmail.assert_called_once()

    @patch('smtplib.SMTP')
    def test_qa_sample_alert(self, mock_smtp, notifier):
        """Test QA sample quality alert."""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        result = notifier.send_qa_sample_alert(
            sample_id="sample_12345",
            recording_session_id="session_67890",
            quality_score=35.0,
            issues=["High noise floor", "Clipping detected", "Low SNR"]
        )

        assert result is True
        mock_server.sendmail.assert_called_once()

    def test_no_recipients_configured(self, notifier):
        """Test behavior when no recipients are configured."""
        notifier.config.recipient_emails = []

        result = notifier.send_notification(
            subject="Test",
            body="Test body"
        )

        assert result is False

    @patch('smtplib.SMTP')
    def test_smtp_connection_failure(self, mock_smtp, notifier):
        """Test handling of SMTP connection failure."""
        mock_smtp.side_effect = smtplib.SMTPException("Connection failed")

        result = notifier.send_notification(
            subject="Test",
            body="Test body"
        )

        assert result is False

    @patch('smtplib.SMTP')
    def test_test_connection_success(self, mock_smtp, notifier):
        """Test SMTP connection test."""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        result = notifier.test_connection()

        assert result is True
        mock_server.ehlo.assert_called()
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()

    def test_validate_config_missing_sender(self):
        """Test configuration validation with missing sender."""
        config = NotificationConfig()
        config.sender_email = ""

        with pytest.raises(ValueError, match="Sender email not configured"):
            GmailNotifier(config)

    def test_validate_config_missing_password(self):
        """Test configuration validation with missing password."""
        config = NotificationConfig()
        config.sender_email = "test@example.com"
        config.sender_password = ""

        with pytest.raises(ValueError, match="Gmail app password not configured"):
            GmailNotifier(config)