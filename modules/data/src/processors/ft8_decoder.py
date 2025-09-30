"""FT8 signal decoder for propagation analysis.

Implements T029: FT8 decoder (FR-024, FR-025).
"""

import asyncio
import logging
import numpy as np
import subprocess
import tempfile
import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

from scipy import signal
from scipy.io import wavfile

from ..config import config
from ..config.frequencies import BAND_CONFIGS
from ..models import SessionLocal, FT8Signal, PropagationRecord

logger = logging.getLogger(__name__)


@dataclass
class FT8Message:
    """Decoded FT8 message."""

    timestamp: datetime
    frequency_hz: float
    snr_db: float
    dt_seconds: float
    message: str
    callsign_hash: Optional[str] = None
    grid_square: Optional[str] = None
    dx_grid: Optional[str] = None
    mode: str = "FT8"


@dataclass
class PropagationReport:
    """FT8-derived propagation report."""

    timestamp: datetime
    band: str
    frequency_khz: float
    tx_grid: str
    rx_grid: str
    distance_km: float
    bearing_degrees: float
    snr_db: float
    mode: str
    propagation_type: str  # F2, Es, MS, etc.


class FT8Decoder:
    """Decoder for FT8 signals with propagation analysis."""

    def __init__(self):
        """Initialize FT8 decoder."""
        self.db = SessionLocal()
        self.ft8_executable = self._find_ft8_decoder()

        # FT8 timing parameters
        self.symbol_period = 12.64  # seconds
        self.sync_tone_spacing = 6.25  # Hz

    def _find_ft8_decoder(self) -> str:
        """Find FT8 decoder executable.

        Returns:
            Path to decoder or error
        """
        # Try common locations for ft8_lib or js8call
        candidates = [
            "/usr/local/bin/ft8_decode",
            "/usr/bin/ft8_decode",
            "ft8_decode",
            "/opt/ft8/ft8_decode",
        ]

        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    logger.info(f"Found FT8 decoder: {candidate}")
                    return candidate
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        logger.warning("No FT8 decoder found, using built-in detection")
        return ""

    async def process_iq_data(
        self,
        iq_data: np.ndarray,
        sample_rate: int,
        center_frequency_khz: float,
        band: str,
        start_time: datetime,
    ) -> List[FT8Message]:
        """Process IQ data for FT8 signals.

        Args:
            iq_data: Complex IQ data
            sample_rate: Sample rate in Hz
            center_frequency_khz: Center frequency
            band: Band name
            start_time: Recording start time

        Returns:
            List of decoded FT8 messages
        """
        logger.info(f"Processing {len(iq_data)} IQ samples for FT8 on {band}")

        # Get band configuration
        band_config = BAND_CONFIGS.get(band)
        if not band_config:
            logger.error(f"Unknown band: {band}")
            return []

        # Extract FT8 frequency window
        ft8_freq_hz = band_config.ft8_freq_khz * 1000
        center_freq_hz = center_frequency_khz * 1000

        # Calculate frequency offset from center
        freq_offset_hz = ft8_freq_hz - center_freq_hz

        # Filter and downsample to FT8 bandwidth
        audio_data = await self._extract_audio_window(
            iq_data, sample_rate, freq_offset_hz, bandwidth_hz=3000
        )

        if audio_data is None:
            return []

        # Process 15-second FT8 intervals
        interval_samples = int(15 * 12000)  # 15 seconds at 12kHz
        messages = []

        for i in range(0, len(audio_data), interval_samples):
            window = audio_data[i : i + interval_samples]
            if len(window) < interval_samples // 2:
                break  # Skip partial windows

            window_start = start_time + timedelta(seconds=i / 12000)

            # Process this 15-second window
            window_messages = await self._decode_ft8_window(
                window, 12000, ft8_freq_hz, window_start
            )
            messages.extend(window_messages)

        # Anonymize callsigns
        for msg in messages:
            msg.callsign_hash = self._anonymize_callsign(msg.message)

        logger.info(f"Decoded {len(messages)} FT8 messages on {band}")
        return messages

    async def _extract_audio_window(
        self,
        iq_data: np.ndarray,
        sample_rate: int,
        freq_offset_hz: float,
        bandwidth_hz: int = 3000,
    ) -> Optional[np.ndarray]:
        """Extract audio window from IQ data.

        Args:
            iq_data: Complex IQ samples
            sample_rate: IQ sample rate
            freq_offset_hz: Frequency offset from center
            bandwidth_hz: Audio bandwidth

        Returns:
            Real audio samples or None
        """
        try:
            # Frequency shift to baseband
            if freq_offset_hz != 0:
                t = np.arange(len(iq_data)) / sample_rate
                shift_signal = np.exp(-1j * 2 * np.pi * freq_offset_hz * t)
                iq_data = iq_data * shift_signal

            # Low-pass filter
            nyquist = sample_rate / 2
            cutoff = min(bandwidth_hz / 2, nyquist * 0.8)
            sos = signal.butter(8, cutoff / nyquist, btype="low", output="sos")
            filtered = signal.sosfilt(sos, iq_data)

            # Decimate to audio rate (12 kHz)
            decimation = sample_rate // 12000
            if decimation > 1:
                audio = signal.decimate(filtered, decimation, ftype="fir")
            else:
                audio = filtered

            # Convert to real (magnitude)
            audio_real = np.abs(audio)

            return audio_real.astype(np.float32)

        except Exception as e:
            logger.error(f"Error extracting audio window: {e}")
            return None

    async def _decode_ft8_window(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        center_freq_hz: float,
        window_start: datetime,
    ) -> List[FT8Message]:
        """Decode FT8 signals in 15-second window.

        Args:
            audio_data: Audio samples
            sample_rate: Audio sample rate
            center_freq_hz: Center frequency
            window_start: Window start time

        Returns:
            List of FT8 messages
        """
        messages = []

        if self.ft8_executable:
            # Use external decoder
            messages = await self._decode_with_external(
                audio_data, sample_rate, center_freq_hz, window_start
            )
        else:
            # Use built-in detection
            messages = await self._detect_ft8_builtin(
                audio_data, sample_rate, center_freq_hz, window_start
            )

        return messages

    async def _decode_with_external(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        center_freq_hz: float,
        window_start: datetime,
    ) -> List[FT8Message]:
        """Decode using external FT8 decoder.

        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            center_freq_hz: Center frequency
            window_start: Window start time

        Returns:
            Decoded messages
        """
        messages = []

        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Convert to 16-bit PCM
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wavfile.write(temp_file.name, sample_rate, audio_int16)

                # Run decoder
                cmd = [
                    self.ft8_executable,
                    "-d",  # Deep decode
                    "-t",
                    window_start.strftime("%Y%m%d_%H%M%S"),
                    temp_file.name,
                ]

                result = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await result.communicate()

                if result.returncode == 0:
                    # Parse decoder output
                    messages = self._parse_decoder_output(
                        stdout.decode(), window_start
                    )
                else:
                    logger.debug(f"Decoder error: {stderr.decode()}")

                # Clean up
                os.unlink(temp_file.name)

        except Exception as e:
            logger.error(f"External decoder error: {e}")

        return messages

    async def _detect_ft8_builtin(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        center_freq_hz: float,
        window_start: datetime,
    ) -> List[FT8Message]:
        """Built-in FT8 signal detection (simplified).

        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            center_freq_hz: Center frequency
            window_start: Window start time

        Returns:
            Detected signals (without full decoding)
        """
        messages = []

        try:
            # FFT analysis to find FT8-like signals
            nperseg = int(sample_rate * 0.16)  # ~160ms windows
            f, t, Sxx = signal.spectrogram(
                audio_data, sample_rate, nperseg=nperseg, noverlap=nperseg // 2
            )

            # Look for signals in FT8 frequency range (200-3000 Hz)
            ft8_freq_mask = (f >= 200) & (f <= 3000)
            ft8_spectrum = Sxx[ft8_freq_mask, :]

            # Detect strong signals
            threshold = np.percentile(ft8_spectrum, 95)
            strong_signals = ft8_spectrum > threshold

            # Find signal peaks
            for freq_idx, time_idx in zip(*np.where(strong_signals)):
                if freq_idx < len(f[ft8_freq_mask]) and time_idx < len(t):
                    frequency = f[ft8_freq_mask][freq_idx]
                    timestamp = window_start + timedelta(seconds=t[time_idx])
                    power = ft8_spectrum[freq_idx, time_idx]

                    # Estimate SNR (very rough)
                    noise_floor = np.median(ft8_spectrum[:, time_idx])
                    snr_db = 10 * np.log10(power / max(noise_floor, 1e-10))

                    if snr_db > -10:  # Minimum SNR threshold
                        messages.append(
                            FT8Message(
                                timestamp=timestamp,
                                frequency_hz=center_freq_hz + frequency,
                                snr_db=snr_db,
                                dt_seconds=0.0,
                                message="[DETECTED]",  # No actual decode
                                mode="FT8_DETECT",
                            )
                        )

        except Exception as e:
            logger.error(f"Built-in detection error: {e}")

        return messages

    def _parse_decoder_output(
        self, output: str, window_start: datetime
    ) -> List[FT8Message]:
        """Parse external decoder output.

        Args:
            output: Decoder stdout
            window_start: Window start time

        Returns:
            Parsed FT8 messages
        """
        messages = []

        # FT8 decoder output format varies, this is a generic parser
        for line in output.strip().split("\n"):
            if not line.strip():
                continue

            try:
                # Common format: HHMMSS SNR DT FREQ MESSAGE
                parts = line.split()
                if len(parts) < 5:
                    continue

                time_str = parts[0]
                snr_db = float(parts[1])
                dt_seconds = float(parts[2])
                freq_hz = float(parts[3])
                message = " ".join(parts[4:])

                # Parse timestamp
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])

                timestamp = window_start.replace(
                    hour=hour, minute=minute, second=second
                )

                # Extract grid squares if present
                grid_match = re.search(r"\b[A-R][A-R]\d\d\b", message)
                grid_square = grid_match.group(0) if grid_match else None

                messages.append(
                    FT8Message(
                        timestamp=timestamp,
                        frequency_hz=freq_hz,
                        snr_db=snr_db,
                        dt_seconds=dt_seconds,
                        message=message,
                        grid_square=grid_square,
                    )
                )

            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse line: {line} ({e})")

        return messages

    def _anonymize_callsign(self, message: str) -> str:
        """Anonymize callsigns in FT8 message.

        Args:
            message: FT8 message text

        Returns:
            SHA256 hash of callsigns
        """
        # Extract callsigns (simplified pattern)
        callsign_pattern = r"\b[A-Z0-9]{1,3}[0-9][A-Z0-9]*\b"
        callsigns = re.findall(callsign_pattern, message)

        if callsigns:
            # Hash the primary callsign
            primary = callsigns[0]
            salt = config.CALLSIGN_SALT or "cascade_salt"
            hash_input = f"{primary}:{salt}".encode()
            return hashlib.sha256(hash_input).hexdigest()[:16]

        return None

    async def store_ft8_signals(
        self, messages: List[FT8Message], session_id: str, band: str
    ):
        """Store FT8 signals in database.

        Args:
            messages: FT8 messages to store
            session_id: Recording session ID
            band: Band name
        """
        try:
            for msg in messages:
                signal_record = FT8Signal(
                    session_id=session_id,
                    timestamp=msg.timestamp,
                    frequency_hz=msg.frequency_hz,
                    snr_db=msg.snr_db,
                    dt_seconds=msg.dt_seconds,
                    message_hash=msg.callsign_hash,
                    grid_square=msg.grid_square,
                    band=band,
                    mode=msg.mode,
                    raw_message=msg.message if config.STORE_RAW_MESSAGES else None,
                )

                self.db.add(signal_record)

            self.db.commit()
            logger.info(f"Stored {len(messages)} FT8 signals for session {session_id}")

        except Exception as e:
            logger.error(f"Error storing FT8 signals: {e}")
            self.db.rollback()

    async def analyze_propagation(
        self, messages: List[FT8Message], band: str
    ) -> List[PropagationReport]:
        """Analyze propagation from FT8 messages.

        Args:
            messages: FT8 messages
            band: Band name

        Returns:
            Propagation reports
        """
        reports = []

        for msg in messages:
            if not msg.grid_square:
                continue

            try:
                # Calculate distance and bearing (simplified)
                # In practice, would use proper grid square calculations
                distance_km = self._estimate_distance(msg.grid_square)
                bearing = self._estimate_bearing(msg.grid_square)

                # Classify propagation mode based on distance and frequency
                prop_type = self._classify_propagation(
                    distance_km, msg.snr_db, band, msg.timestamp
                )

                report = PropagationReport(
                    timestamp=msg.timestamp,
                    band=band,
                    frequency_khz=msg.frequency_hz / 1000,
                    tx_grid=msg.grid_square,
                    rx_grid="XX00",  # Would be set from receiver location
                    distance_km=distance_km,
                    bearing_degrees=bearing,
                    snr_db=msg.snr_db,
                    mode="FT8",
                    propagation_type=prop_type,
                )

                reports.append(report)

            except Exception as e:
                logger.debug(f"Error analyzing propagation for {msg.message}: {e}")

        return reports

    def _estimate_distance(self, grid_square: str) -> float:
        """Estimate distance from grid square (placeholder).

        Args:
            grid_square: 4-character grid square

        Returns:
            Distance in kilometers
        """
        # Placeholder - would calculate actual distance
        return 1000.0 + hash(grid_square) % 5000

    def _estimate_bearing(self, grid_square: str) -> float:
        """Estimate bearing to grid square (placeholder).

        Args:
            grid_square: 4-character grid square

        Returns:
            Bearing in degrees
        """
        # Placeholder - would calculate actual bearing
        return hash(grid_square) % 360

    def _classify_propagation(
        self, distance_km: float, snr_db: float, band: str, timestamp: datetime
    ) -> str:
        """Classify propagation mode.

        Args:
            distance_km: Distance
            snr_db: Signal strength
            band: Band
            timestamp: Time

        Returns:
            Propagation type
        """
        # Simplified classification
        if distance_km < 300:
            return "NVIS"
        elif distance_km < 800:
            return "F2_SHORT"
        elif distance_km < 3000:
            return "F2_MEDIUM"
        elif distance_km < 8000:
            return "F2_LONG"
        else:
            return "F2_DX"

    def close(self):
        """Close decoder resources."""
        if self.db:
            self.db.close()


async def process_ft8_file(file_path: str) -> Dict[str, Any]:
    """Process FT8 from audio file (for testing).

    Args:
        file_path: Path to audio file

    Returns:
        Processing results
    """
    decoder = FT8Decoder()

    try:
        # Load audio file
        sample_rate, audio_data = wavfile.read(file_path)

        # Convert to complex IQ if needed
        if audio_data.dtype != np.complex64:
            iq_data = audio_data.astype(np.float32)
            if len(iq_data.shape) == 2:
                iq_data = iq_data[:, 0] + 1j * iq_data[:, 1]
            else:
                iq_data = iq_data + 1j * np.zeros_like(iq_data)
        else:
            iq_data = audio_data

        # Process
        messages = await decoder.process_iq_data(
            iq_data, sample_rate, 14080, "20m", datetime.utcnow()
        )

        return {
            "file": file_path,
            "sample_rate": sample_rate,
            "duration_seconds": len(iq_data) / sample_rate,
            "messages_decoded": len(messages),
            "messages": [asdict(msg) for msg in messages[:10]],  # First 10
        }

    finally:
        decoder.close()