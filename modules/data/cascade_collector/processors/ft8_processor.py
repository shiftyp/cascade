"""FT8 signal processor for CASCADE."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FT8Message:
    """Decoded FT8 message."""
    timestamp: datetime
    frequency: float
    snr: float
    dt: float  # Time offset
    call1: str
    call2: str
    grid: Optional[str]
    power: Optional[int]
    raw_message: str


class FT8Processor:
    """Processes FT8 signals from IQ data."""

    def __init__(self):
        """Initialize FT8 processor."""
        self.messages_decoded = 0
        self.total_samples_processed = 0
        self.decoder_available = False
        self._check_decoder()

    def _check_decoder(self):
        """Check if FT8 decoder is available."""
        # In production, would check for ft8_lib or jt9 binary
        # For now, we'll simulate
        self.decoder_available = True
        logger.info(f"FT8 decoder available: {self.decoder_available}")

    async def process_recording(self, audio_data: np.ndarray,
                               sample_rate: int,
                               frequency: float) -> List[FT8Message]:
        """Process audio recording for FT8 signals.

        Args:
            audio_data: Audio samples (IQ or real)
            sample_rate: Sample rate in Hz
            frequency: Center frequency in Hz

        Returns:
            List of decoded FT8 messages
        """
        if not self.decoder_available:
            logger.warning("FT8 decoder not available")
            return []

        # Update stats
        self.total_samples_processed += len(audio_data)

        # Simulate FT8 decoding (in production would use actual decoder)
        messages = self._simulate_ft8_decode(audio_data, sample_rate, frequency)

        # Anonymize callsigns
        for msg in messages:
            msg.call1 = self._anonymize_callsign(msg.call1)
            msg.call2 = self._anonymize_callsign(msg.call2)

        self.messages_decoded += len(messages)
        logger.info(f"Decoded {len(messages)} FT8 messages from recording")

        return messages

    def _simulate_ft8_decode(self, audio_data: np.ndarray,
                            sample_rate: int,
                            frequency: float) -> List[FT8Message]:
        """Simulate FT8 decoding for testing.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            frequency: Center frequency

        Returns:
            List of simulated FT8 messages
        """
        messages = []

        # Generate some realistic-looking test messages
        test_patterns = [
            ("CQ DX W1ABC FN42", -10, 0.2),
            ("K2DEF W1ABC -06", -6, 0.3),
            ("W1ABC K2DEF R-08", -8, 0.4),
            ("K2DEF W1ABC RR73", -5, 0.5),
            ("CQ NA VE3GHI FN25", -12, 0.1),
        ]

        # Simulate detection based on signal strength
        signal_power = np.std(audio_data)
        if signal_power > 0.01:  # Arbitrary threshold
            for pattern, snr, dt in test_patterns[:2]:  # Return 2 messages
                parts = pattern.split()

                msg = FT8Message(
                    timestamp=datetime.now(timezone.utc),
                    frequency=frequency,
                    snr=snr + np.random.randn() * 2,
                    dt=dt + np.random.randn() * 0.1,
                    call1=parts[1] if len(parts) > 1 else "UNKNOWN",
                    call2=parts[2] if len(parts) > 2 else "UNKNOWN",
                    grid=parts[3] if len(parts) > 3 and self._is_grid(parts[3]) else None,
                    power=None,
                    raw_message=pattern
                )
                messages.append(msg)

        return messages

    def _is_grid(self, text: str) -> bool:
        """Check if text is a valid grid square.

        Args:
            text: Text to check

        Returns:
            True if valid grid square
        """
        if len(text) not in [4, 6]:
            return False

        if len(text) >= 4:
            # Check first two are letters A-R
            if not (text[0].upper() in "ABCDEFGHIJKLMNOPQR" and
                   text[1].upper() in "ABCDEFGHIJKLMNOPQR"):
                return False
            # Check next two are digits
            if not (text[2].isdigit() and text[3].isdigit()):
                return False

        if len(text) == 6:
            # Check last two are letters
            if not (text[4].isalpha() and text[5].isalpha()):
                return False

        return True

    def _anonymize_callsign(self, callsign: str) -> str:
        """Anonymize callsign using one-way hash.

        Args:
            callsign: Original callsign

        Returns:
            Anonymized version
        """
        if callsign in ["CQ", "QRZ", "DE", "UNKNOWN"]:
            return callsign

        # Create hash
        hash_obj = hashlib.sha256(callsign.encode())
        hash_hex = hash_obj.hexdigest()

        # Return first 8 chars as anonymous ID
        return f"ANON_{hash_hex[:8].upper()}"

    def extract_propagation_data(self, messages: List[FT8Message]) -> Dict[str, Any]:
        """Extract propagation data from FT8 messages.

        Args:
            messages: List of FT8 messages

        Returns:
            Propagation analysis data
        """
        if not messages:
            return {}

        snr_values = [msg.snr for msg in messages]
        dt_values = [msg.dt for msg in messages]

        propagation_data = {
            "message_count": len(messages),
            "avg_snr": np.mean(snr_values),
            "min_snr": np.min(snr_values),
            "max_snr": np.max(snr_values),
            "avg_dt": np.mean(dt_values),
            "unique_stations": len(set(msg.call1 for msg in messages) |
                                 set(msg.call2 for msg in messages)),
            "grid_squares": list(set(msg.grid for msg in messages if msg.grid))
        }

        return propagation_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "messages_decoded": self.messages_decoded,
            "samples_processed": self.total_samples_processed,
            "decoder_available": self.decoder_available,
            "avg_messages_per_mb": self.messages_decoded / max(1,
                self.total_samples_processed / 1024 / 1024)
        }