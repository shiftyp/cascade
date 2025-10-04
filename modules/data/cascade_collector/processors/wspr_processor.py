"""WSPR signal processor for CASCADE."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class WSPRMessage:
    """Decoded WSPR message."""
    timestamp: datetime
    frequency: float
    snr: float
    dt: float  # Time offset
    drift: float  # Frequency drift Hz/min
    callsign: str
    grid: str
    power: int  # dBm
    raw_message: str


class WSPRProcessor:
    """Processes WSPR signals from IQ data."""

    def __init__(self):
        """Initialize WSPR processor."""
        self.messages_decoded = 0
        self.total_samples_processed = 0
        self.decoder_available = False
        self._check_decoder()

    def _check_decoder(self):
        """Check if WSPR decoder is available."""
        # In production, would check for wsprd binary
        self.decoder_available = True
        logger.info(f"WSPR decoder available: {self.decoder_available}")

    async def process_recording(self, audio_data: np.ndarray,
                               sample_rate: int,
                               frequency: float) -> List[WSPRMessage]:
        """Process audio recording for WSPR signals.

        Args:
            audio_data: Audio samples (IQ or real)
            sample_rate: Sample rate in Hz
            frequency: Center frequency in Hz

        Returns:
            List of decoded WSPR messages
        """
        if not self.decoder_available:
            logger.warning("WSPR decoder not available")
            return []

        # WSPR requires 2-minute recordings
        expected_samples = sample_rate * 120  # 2 minutes
        if len(audio_data) < expected_samples * 0.9:
            logger.warning(f"Recording too short for WSPR: {len(audio_data)} samples")
            return []

        # Update stats
        self.total_samples_processed += len(audio_data)

        # Simulate WSPR decoding
        messages = self._simulate_wspr_decode(audio_data, sample_rate, frequency)

        # Anonymize callsigns
        for msg in messages:
            msg.callsign = self._anonymize_callsign(msg.callsign)

        self.messages_decoded += len(messages)
        logger.info(f"Decoded {len(messages)} WSPR messages from recording")

        return messages

    def _simulate_wspr_decode(self, audio_data: np.ndarray,
                             sample_rate: int,
                             frequency: float) -> List[WSPRMessage]:
        """Simulate WSPR decoding for testing.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            frequency: Center frequency

        Returns:
            List of simulated WSPR messages
        """
        messages = []

        # WSPR test patterns (callsign, grid, power_dBm)
        test_patterns = [
            ("W1ABC", "FN42", 23),
            ("G0XYZ", "IO91", 37),
            ("VK2DEF", "QF56", 30),
            ("JA1GHI", "PM95", 33),
        ]

        # Simulate detection
        signal_power = np.std(audio_data)
        if signal_power > 0.005:  # Lower threshold for WSPR
            for callsign, grid, power in test_patterns[:2]:
                msg = WSPRMessage(
                    timestamp=datetime.now(timezone.utc),
                    frequency=frequency,
                    snr=-20 + np.random.randn() * 5,  # WSPR works at very low SNR
                    dt=np.random.randn() * 2,
                    drift=np.random.randn() * 0.5,
                    callsign=callsign,
                    grid=grid,
                    power=power,
                    raw_message=f"{callsign} {grid} {power}"
                )
                messages.append(msg)

        return messages

    def _anonymize_callsign(self, callsign: str) -> str:
        """Anonymize callsign using one-way hash.

        Args:
            callsign: Original callsign

        Returns:
            Anonymized version
        """
        if not callsign:
            return "UNKNOWN"

        # Create hash
        hash_obj = hashlib.sha256(callsign.encode())
        hash_hex = hash_obj.hexdigest()

        # Return first 8 chars as anonymous ID
        return f"WSPR_{hash_hex[:8].upper()}"

    def extract_propagation_data(self, messages: List[WSPRMessage]) -> Dict[str, Any]:
        """Extract propagation data from WSPR messages.

        Args:
            messages: List of WSPR messages

        Returns:
            Propagation analysis data
        """
        if not messages:
            return {}

        snr_values = [msg.snr for msg in messages]
        power_values = [msg.power for msg in messages]
        drift_values = [msg.drift for msg in messages]

        propagation_data = {
            "message_count": len(messages),
            "avg_snr": np.mean(snr_values),
            "min_snr": np.min(snr_values),
            "max_snr": np.max(snr_values),
            "avg_power_dbm": np.mean(power_values),
            "avg_drift_hz": np.mean(drift_values),
            "unique_stations": len(set(msg.callsign for msg in messages)),
            "grid_squares": list(set(msg.grid for msg in messages)),
            "path_loss_estimates": self._calculate_path_loss(messages)
        }

        return propagation_data

    def _calculate_path_loss(self, messages: List[WSPRMessage]) -> List[Dict[str, Any]]:
        """Calculate path loss estimates from WSPR data.

        Args:
            messages: WSPR messages

        Returns:
            Path loss estimates
        """
        estimates = []
        for msg in messages:
            # Simplified path loss calculation
            # Real implementation would use actual distance calculation
            estimated_loss = msg.power - msg.snr - 10  # Simplified
            estimates.append({
                "grid": msg.grid,
                "power_dbm": msg.power,
                "snr": msg.snr,
                "estimated_loss_db": estimated_loss
            })
        return estimates

    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "messages_decoded": self.messages_decoded,
            "samples_processed": self.total_samples_processed,
            "decoder_available": self.decoder_available,
            "avg_messages_per_recording": self.messages_decoded / max(1,
                self.total_samples_processed / (12000 * 120))  # Per 2-min recording
        }