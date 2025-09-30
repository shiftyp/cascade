"""Gmail notification service for CASCADE data collection alerts."""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import os
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """Gmail notification configuration."""

    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipient_emails: List[str] = None
    enable_ssl: bool = True
    enable_tls: bool = True

    def __post_init__(self):
        """Initialize from environment variables if not set."""
        if not self.sender_email:
            self.sender_email = os.environ.get("GMAIL_SENDER_EMAIL", "")
        if not self.sender_password:
            self.sender_password = os.environ.get("GMAIL_APP_PASSWORD", "")
        if not self.recipient_emails:
            recipients = os.environ.get("NOTIFICATION_RECIPIENTS", "")
            self.recipient_emails = [r.strip() for r in recipients.split(",") if r.strip()]


class GmailNotifier:
    """Sends email notifications for CASCADE collection events."""

    def __init__(self, config: Optional[NotificationConfig] = None):
        """Initialize Gmail notifier with configuration.

        Args:
            config: Notification configuration (uses env vars if None)
        """
        self.config = config or NotificationConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate notification configuration."""
        if not self.config.sender_email:
            raise ValueError("Sender email not configured (set GMAIL_SENDER_EMAIL)")
        if not self.config.sender_password:
            raise ValueError("Gmail app password not configured (set GMAIL_APP_PASSWORD)")
        if not self.config.recipient_emails:
            logger.warning("No recipient emails configured")

    def send_notification(
        self,
        subject: str,
        body: str,
        priority: str = "normal",
        html_body: Optional[str] = None,
        attachments: Optional[Dict[str, bytes]] = None
    ) -> bool:
        """Send email notification.

        Args:
            subject: Email subject line
            body: Plain text email body
            priority: Email priority (low, normal, high)
            html_body: Optional HTML version of email body
            attachments: Optional dict of filename -> bytes content

        Returns:
            True if notification sent successfully
        """
        if not self.config.recipient_emails:
            logger.warning("No recipients configured, skipping notification")
            return False

        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.config.sender_email
            message["To"] = ", ".join(self.config.recipient_emails)

            # Set priority header
            if priority == "high":
                message["X-Priority"] = "1"
                message["Importance"] = "high"
            elif priority == "low":
                message["X-Priority"] = "5"
                message["Importance"] = "low"

            # Add plain text part
            text_part = MIMEText(body, "plain")
            message.attach(text_part)

            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, "html")
                message.attach(html_part)

            # Send email
            context = ssl.create_default_context()

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.enable_tls:
                    server.starttls(context=context)
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(
                    self.config.sender_email,
                    self.config.recipient_emails,
                    message.as_string()
                )

            logger.info(f"Notification sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    def send_space_weather_alert(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """Send space weather event notification.

        Args:
            event_type: Type of space weather event (flare, storm, etc)
            event_data: Event details including xray_class, flux, time

        Returns:
            True if notification sent successfully
        """
        subject = f"🌟 CASCADE: Space Weather Alert - {event_type}"

        # Format event details
        lines = [
            f"Space Weather Event Detected",
            f"Type: {event_type}",
            f"Time: {event_data.get('time', datetime.utcnow().isoformat())}",
            "",
            "Event Details:"
        ]

        for key, value in event_data.items():
            if key != "time":
                lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "Collection system is scaling up to capture enhanced propagation.",
            f"Current active collectors: {event_data.get('active_collectors', 'unknown')}",
            f"Target collectors: {event_data.get('target_collectors', 20)}",
        ])

        body = "\n".join(lines)

        # Determine priority based on event severity
        xray_class = event_data.get("xray_class", "")
        priority = "high" if xray_class.startswith(("X", "M")) else "normal"

        return self.send_notification(subject, body, priority=priority)

    def send_collection_status(
        self,
        daily_hours: float,
        total_hours: float,
        active_sdrs: int,
        failed_sdrs: List[str],
        storage_used_gb: float
    ) -> bool:
        """Send daily collection status report.

        Args:
            daily_hours: Hours collected in last 24h
            total_hours: Total hours collected so far
            active_sdrs: Number of currently active SDRs
            failed_sdrs: List of failed SDR names
            storage_used_gb: Storage space used in GB

        Returns:
            True if notification sent successfully
        """
        subject = f"CASCADE: Daily Status - {daily_hours:.1f} hours collected"

        lines = [
            f"CASCADE Data Collection Status Report",
            f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}",
            "",
            "Collection Metrics:",
            f"  Last 24 hours: {daily_hours:.1f} hours",
            f"  Total collected: {total_hours:,.1f} hours",
            f"  Collection rate: {daily_hours:.1f} hours/day",
            f"  Progress: {(total_hours / 200000) * 100:.1f}% of target",
            "",
            "SDR Status:",
            f"  Active SDRs: {active_sdrs}",
            f"  Failed SDRs: {len(failed_sdrs)}",
        ]

        if failed_sdrs:
            lines.append("  Failed: " + ", ".join(failed_sdrs[:5]))
            if len(failed_sdrs) > 5:
                lines.append(f"    ... and {len(failed_sdrs) - 5} more")

        lines.extend([
            "",
            "Storage:",
            f"  Used: {storage_used_gb:,.1f} GB",
            f"  Estimated remaining: {(75000 - storage_used_gb):,.1f} GB",
            "",
            "System Health: " + ("✅ Healthy" if active_sdrs >= 6 else "⚠️ Below minimum"),
        ])

        body = "\n".join(lines)
        priority = "high" if active_sdrs < 6 else "normal"

        return self.send_notification(subject, body, priority=priority)

    def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        affected_components: List[str],
        suggested_action: Optional[str] = None
    ) -> bool:
        """Send error alert notification.

        Args:
            error_type: Type of error (connection, storage, processing, etc)
            error_message: Detailed error message
            affected_components: List of affected system components
            suggested_action: Optional suggested remediation action

        Returns:
            True if notification sent successfully
        """
        subject = f"⚠️ CASCADE: Error Alert - {error_type}"

        lines = [
            f"CASCADE System Error Detected",
            f"Time: {datetime.utcnow().isoformat()}",
            f"Type: {error_type}",
            "",
            "Error Details:",
            error_message,
            "",
            "Affected Components:",
        ]

        for component in affected_components:
            lines.append(f"  - {component}")

        if suggested_action:
            lines.extend([
                "",
                "Suggested Action:",
                suggested_action
            ])

        lines.extend([
            "",
            "Please investigate immediately to minimize data loss.",
        ])

        body = "\n".join(lines)

        return self.send_notification(subject, body, priority="high")

    def send_qa_sample_alert(
        self,
        sample_id: str,
        recording_session_id: str,
        quality_score: float,
        issues: List[str]
    ) -> bool:
        """Send QA sample quality alert.

        Args:
            sample_id: ID of the QA sample
            recording_session_id: Associated recording session
            quality_score: Quality score (0-100)
            issues: List of quality issues detected

        Returns:
            True if notification sent successfully
        """
        subject = f"CASCADE: QA Alert - Sample {sample_id[:8]} quality issue"

        lines = [
            f"QA Sample Quality Alert",
            f"Sample ID: {sample_id}",
            f"Session: {recording_session_id}",
            f"Quality Score: {quality_score:.1f}/100",
            "",
            "Issues Detected:",
        ]

        for issue in issues:
            lines.append(f"  - {issue}")

        lines.extend([
            "",
            "This sample has been quarantined for review.",
            "Access the QA dashboard to investigate further.",
        ])

        body = "\n".join(lines)
        priority = "high" if quality_score < 50 else "normal"

        return self.send_notification(subject, body, priority=priority)

    def test_connection(self) -> bool:
        """Test Gmail SMTP connection and credentials.

        Returns:
            True if connection successful
        """
        try:
            context = ssl.create_default_context()

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.ehlo()
                if self.config.enable_tls:
                    server.starttls(context=context)
                    server.ehlo()
                server.login(self.config.sender_email, self.config.sender_password)

            logger.info("Gmail SMTP connection test successful")
            return True

        except Exception as e:
            logger.error(f"Gmail SMTP connection test failed: {e}")
            return False


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Gmail notifications")
    parser.add_argument("--test", action="store_true", help="Send test notification")
    parser.add_argument("--check", action="store_true", help="Test connection only")
    parser.add_argument("--to", help="Override recipient email")

    args = parser.parse_args()

    # Create notifier
    config = NotificationConfig()
    if args.to:
        config.recipient_emails = [args.to]

    notifier = GmailNotifier(config)

    if args.check:
        success = notifier.test_connection()
        print("Connection test:", "✅ Success" if success else "❌ Failed")

    elif args.test:
        success = notifier.send_notification(
            subject="CASCADE Test Notification",
            body="This is a test notification from the CASCADE data collection system.\n\n"
                 "If you received this email, notifications are working correctly.",
            priority="normal"
        )
        print("Test notification:", "✅ Sent" if success else "❌ Failed")

    else:
        parser.print_help()